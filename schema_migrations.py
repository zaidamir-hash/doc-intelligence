"""Small checksum-tracked PostgreSQL migration runner for Lexis."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine, text


MIGRATIONS_DIRECTORY = Path(__file__).parent / "migrations"
MIGRATION_NAME_PATTERN = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


@dataclass(frozen=True)
class Migration:
    version: str
    name: str
    path: Path
    checksum: str
    sql: str


def discover_migrations(directory: Path = MIGRATIONS_DIRECTORY) -> list[Migration]:
    """Return valid migration files in deterministic version order."""

    migrations: list[Migration] = []
    seen_versions: set[str] = set()
    for path in sorted(directory.glob("*.sql")):
        match = MIGRATION_NAME_PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(f"Invalid migration filename: {path.name}")
        version = match.group(1)
        if version in seen_versions:
            raise ValueError(f"Duplicate migration version: {version}")
        seen_versions.add(version)
        sql = path.read_text(encoding="utf-8")
        if not sql.strip():
            raise ValueError(f"Migration is empty: {path.name}")
        migrations.append(
            Migration(
                version=version,
                name=path.name,
                path=path,
                checksum=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
                sql=sql,
            )
        )
    return migrations


def _ensure_history_table(connection) -> None:
    connection.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version varchar(4) PRIMARY KEY,
                name text NOT NULL,
                checksum char(64) NOT NULL,
                applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
    )


def migration_status(engine: Engine) -> list[dict[str, str | bool]]:
    migrations = discover_migrations()
    with engine.begin() as connection:
        _ensure_history_table(connection)
        applied = {
            row.version: row.checksum
            for row in connection.execute(
                text("SELECT version, checksum FROM schema_migrations")
            )
        }
    return [
        {
            "version": migration.version,
            "name": migration.name,
            "applied": migration.version in applied,
            "checksum_matches": applied.get(migration.version) in {
                None,
                migration.checksum,
            },
        }
        for migration in migrations
    ]


def apply_migrations(engine: Engine) -> list[str]:
    """Apply pending migrations atomically and reject edited history."""

    migrations = discover_migrations()
    applied_now: list[str] = []
    with engine.begin() as connection:
        _ensure_history_table(connection)
        connection.execute(
            text("SELECT pg_advisory_xact_lock(hashtext('lexis-schema-migrations'))")
        )
        applied = {
            row.version: row.checksum
            for row in connection.execute(
                text("SELECT version, checksum FROM schema_migrations")
            )
        }
        for migration in migrations:
            existing_checksum = applied.get(migration.version)
            if existing_checksum:
                if existing_checksum != migration.checksum:
                    raise RuntimeError(
                        f"Applied migration {migration.version} checksum changed"
                    )
                continue
            cursor = connection.connection.cursor()
            try:
                cursor.execute(migration.sql)
            finally:
                cursor.close()
            connection.execute(
                text(
                    """
                    INSERT INTO schema_migrations (version, name, checksum)
                    VALUES (:version, :name, :checksum)
                    """
                ),
                {
                    "version": migration.version,
                    "name": migration.name,
                    "checksum": migration.checksum,
                },
            )
            applied_now.append(migration.version)
    return applied_now
