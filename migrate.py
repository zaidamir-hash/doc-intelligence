"""Apply or inspect Lexis database schema migrations."""

from __future__ import annotations

import argparse

from database import engine
from schema_migrations import apply_migrations, migration_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("upgrade", "status"), nargs="?", default="upgrade"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "status":
        for migration in migration_status(engine):
            state = "applied" if migration["applied"] else "pending"
            checksum = "ok" if migration["checksum_matches"] else "CHANGED"
            print(
                f"{migration['version']} {migration['name']}: "
                f"{state}, checksum={checksum}"
            )
        return 0

    applied = apply_migrations(engine)
    if applied:
        print(f"Applied migrations: {', '.join(applied)}")
    else:
        print("Database schema is already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
