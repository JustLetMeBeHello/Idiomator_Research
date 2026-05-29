"""
One-shot migration: copy all rows from local SQLite annotations.db → Railway Postgres.

Usage:
    DATABASE_URL=postgresql://... python migrate_sqlite_to_postgres.py

Set DATABASE_URL to the Railway Postgres connection string before running.
Safe to run multiple times — uses INSERT OR IGNORE / ON CONFLICT DO NOTHING.
"""

import os
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent
SQLITE_PATH = HERE / "annotations.db"

pg_url = os.environ.get("DATABASE_URL", "")
if not pg_url or "sqlite" in pg_url:
    raise SystemExit("Set DATABASE_URL to the Railway Postgres URL before running.")

pg_url = pg_url.replace("postgres://", "postgresql://", 1)

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, declarative_base
from main import AnnotationDB, Base

# Source: local SQLite
sqlite_engine = sa.create_engine(f"sqlite:///{SQLITE_PATH}", connect_args={"check_same_thread": False})
SqliteSession = sessionmaker(bind=sqlite_engine)

# Destination: Railway Postgres
pg_engine = sa.create_engine(pg_url)
Base.metadata.create_all(bind=pg_engine)
PgSession = sessionmaker(bind=pg_engine)

src = SqliteSession()
dst = PgSession()

rows = src.query(AnnotationDB).all()
print(f"Migrating {len(rows)} rows...")

copied = skipped = 0
for row in rows:
    existing = dst.get(AnnotationDB, row.annotation_id)
    if existing:
        skipped += 1
        continue
    dst.add(AnnotationDB(**{
        c.key: getattr(row, c.key)
        for c in sa.inspect(AnnotationDB).mapper.column_attrs
    }))
    copied += 1

dst.commit()
src.close()
dst.close()

print(f"Done. Copied: {copied}, skipped (already present): {skipped}")

# Verify
dst2 = PgSession()
n = dst2.query(AnnotationDB).count()
dst2.close()
print(f"Postgres row count after migration: {n}")
