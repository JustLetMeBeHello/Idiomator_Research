"""
IdiomBank Annotation API

Endpoints:
  GET  /                        health check
  GET  /test/languages          list available languages + example counts
  GET  /test/{language}         serve enriched test JSONL for a language
  GET  /annotations             list all annotators + progress
  GET  /export/{annotator}      download annotator's annotations as JSONL
  POST /save                    upsert one annotation record
"""

import io
import json as _json
import os
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean, text
)
from sqlalchemy.orm import declarative_base, sessionmaker


# ── Paths ──────────────────────────────────────────────────────────────────────

HERE     = Path(__file__).parent
DATA_DIR = HERE / "data"
DB_PATH  = HERE / "annotations.db"


# ── Database ───────────────────────────────────────────────────────────────────

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class AnnotationDB(Base):
    __tablename__ = "annotations"

    annotation_id            = Column(String,  primary_key=True)
    meaning_id               = Column(String,  index=True)
    idiom_id                 = Column(String)
    annotator                = Column(String,  index=True)
    session_id               = Column(String)
    language                 = Column(String,  index=True)
    idiom                    = Column(String)
    sentence                 = Column(String)
    original_idiomaticity    = Column(String)
    idiomaticity_verdict     = Column(String)
    label_changed            = Column(Boolean)
    span_start               = Column(Integer)
    span_end                 = Column(Integer)
    matched_span             = Column(String)
    span_correct             = Column(Boolean)
    span_correction          = Column(String)
    sense_correct            = Column(String)   # "correct" | "wrong" | "uncertain" | null
    sense_notes              = Column(String)
    notes                    = Column(String)
    annotation_time_seconds  = Column(Float)
    validated_at             = Column(String)


Base.metadata.create_all(bind=engine)

# Migrate older databases that lack the new columns
def _migrate():
    with engine.connect() as conn:
        for col, typedef in [("sense_correct", "TEXT"), ("sense_notes", "TEXT")]:
            try:
                conn.execute(text(f"ALTER TABLE annotations ADD COLUMN {col} {typedef}"))
                conn.commit()
            except Exception:
                pass  # column already exists

_migrate()


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="IdiomBank Annotation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schema ────────────────────────────────────────────────────────────

class Annotation(BaseModel):
    annotation_id:           str
    meaning_id:              str
    idiom_id:                str
    annotator:               str
    session_id:              str
    language:                str
    idiom:                   str
    sentence:                str
    original_idiomaticity:   str
    idiomaticity_verdict:    str
    label_changed:           bool
    span_start:              int
    span_end:                int
    matched_span:            str
    span_correct:            bool
    span_correction:         str | None = None
    sense_correct:           str | None = None   # "correct" | "wrong" | "uncertain"
    sense_notes:             str | None = None
    notes:                   str | None = None
    annotation_time_seconds: float
    validated_at:            str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_test(language: str) -> list[dict]:
    path = DATA_DIR / language / "test.jsonl"
    if not path.exists():
        raise HTTPException(404, f"No test data for language '{language}'")
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(_json.loads(line))
    return examples


def _row_to_dict(r: AnnotationDB) -> dict:
    return {
        "annotation_id":          r.annotation_id,
        "meaning_id":             r.meaning_id,
        "idiom_id":               r.idiom_id,
        "annotator":              r.annotator,
        "session_id":             r.session_id,
        "language":               r.language,
        "idiom":                  r.idiom,
        "sentence":               r.sentence,
        "original_idiomaticity":  r.original_idiomaticity,
        "idiomaticity_verdict":   r.idiomaticity_verdict,
        "label_changed":          r.label_changed,
        "span_start":             r.span_start,
        "span_end":               r.span_end,
        "matched_span":           r.matched_span,
        "span_correct":           r.span_correct,
        "span_correction":        r.span_correction,
        "sense_correct":          r.sense_correct,
        "sense_notes":            r.sense_notes,
        "notes":                  r.notes,
        "annotation_time_seconds": r.annotation_time_seconds,
        "validated_at":           r.validated_at,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "running", "service": "IdiomBank Annotation API"}


@app.get("/test/languages")
def list_languages():
    """Return available languages, example counts, and per-annotator completion."""
    languages = {}
    if DATA_DIR.exists():
        for lang_dir in sorted(DATA_DIR.iterdir()):
            if not lang_dir.is_dir():
                continue
            test_file = lang_dir / "test.jsonl"
            if not test_file.exists():
                continue
            count = sum(1 for line in open(test_file, encoding="utf-8") if line.strip())
            languages[lang_dir.name] = {"total": count}

    db = SessionLocal()
    try:
        rows = db.query(
            AnnotationDB.language,
            AnnotationDB.annotator,
            AnnotationDB.meaning_id,
        ).all()
        by_lang: dict = defaultdict(lambda: defaultdict(set))
        for lang, ann, mid in rows:
            by_lang[lang][ann].add(mid)
        for lang, ann_dict in by_lang.items():
            if lang in languages:
                languages[lang]["annotators"] = {
                    ann: len(mids) for ann, mids in ann_dict.items()
                }
    finally:
        db.close()

    return languages


@app.get("/test/{language}")
def get_test_data(language: str):
    """Serve the enriched test JSONL for a language."""
    examples = _load_test(language)
    lines = "\n".join(_json.dumps(ex, ensure_ascii=False) for ex in examples)
    filename = f"{language.lower()}_test.jsonl"
    return StreamingResponse(
        io.StringIO(lines),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/annotations")
def list_annotators():
    """Return all annotators and their annotation counts by language."""
    db = SessionLocal()
    try:
        rows = db.query(AnnotationDB.annotator, AnnotationDB.language).all()
        counts: dict = {}
        for annotator, lang in rows:
            if annotator not in counts:
                counts[annotator] = {"total": 0, "by_language": {}}
            counts[annotator]["total"] += 1
            counts[annotator]["by_language"][lang] = (
                counts[annotator]["by_language"].get(lang, 0) + 1
            )
        return counts
    finally:
        db.close()


@app.get("/export/{annotator}")
def export_annotator(annotator: str, lang: str | None = None):
    """Download a specific annotator's annotations as JSONL."""
    db = SessionLocal()
    try:
        q = db.query(AnnotationDB).filter(AnnotationDB.annotator == annotator)
        if lang:
            q = q.filter(AnnotationDB.language == lang)
        rows = q.all()
        if not rows:
            raise HTTPException(
                404,
                f"No annotations for '{annotator}'"
                + (f" / language '{lang}'" if lang else ""),
            )
        lines = "\n".join(_json.dumps(_row_to_dict(r), ensure_ascii=False) for r in rows)
        fname = f"{annotator}_annotations{'_' + lang if lang else ''}.jsonl"
        return StreamingResponse(
            io.StringIO(lines),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )
    finally:
        db.close()


@app.post("/save")
def save_annotation(annotation: Annotation):
    """Upsert one annotation record."""
    db = SessionLocal()
    try:
        row = AnnotationDB(**annotation.model_dump())
        db.merge(row)
        db.commit()
        return {"status": "saved", "annotation_id": annotation.annotation_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(500, str(e))
    finally:
        db.close()
