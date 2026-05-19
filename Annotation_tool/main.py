from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    Boolean
)

from sqlalchemy.orm import declarative_base, sessionmaker


# ── Database Setup ─────────────────────────────────────────────

DATABASE_URL = "sqlite:///./annotations.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


# ── FastAPI App ────────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Database Model ─────────────────────────────────────────────

class AnnotationDB(Base):
    __tablename__ = "annotations"

    annotation_id = Column(String, primary_key=True)

    meaning_id = Column(String)
    idiom_id = Column(String)

    annotator = Column(String)
    session_id = Column(String)

    idiom = Column(String)
    language = Column(String)
    sentence = Column(String)

    original_idiomaticity = Column(String)
    idiomaticity_verdict = Column(String)

    label_changed = Column(Boolean)

    span_start = Column(Integer)
    span_end = Column(Integer)

    matched_span = Column(String)

    span_correct = Column(Boolean)
    span_correction = Column(String)

    notes = Column(String)

    annotation_time_seconds = Column(Float)

    validated_at = Column(String)


# Create DB tables
Base.metadata.create_all(bind=engine)


# ── API Schema ─────────────────────────────────────────────────

class Annotation(BaseModel):
    annotation_id: str

    meaning_id: str
    idiom_id: str

    annotator: str
    session_id: str

    idiom: str
    language: str
    sentence: str

    original_idiomaticity: str
    idiomaticity_verdict: str

    label_changed: bool

    span_start: int
    span_end: int

    matched_span: str

    span_correct: bool
    span_correction: str | None = None

    notes: str | None = None

    annotation_time_seconds: float

    validated_at: str


# ── Routes ─────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "IdiomBank Annotation API"
    }


@app.post("/save")
def save_annotation(annotation: Annotation):
    db = SessionLocal()

    try:
        row = AnnotationDB(**annotation.dict())

        # merge() upserts by primary key — safe if the same annotation_id
        # is re-submitted (e.g. annotator edits a previous answer)
        db.merge(row)
        db.commit()

        return {
            "status": "saved",
            "annotation_id": annotation.annotation_id
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        db.close()