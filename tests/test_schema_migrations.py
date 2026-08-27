from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from schema_migrations import discover_migrations


class MigrationDiscoveryTests(unittest.TestCase):
    def test_discovers_versioned_sql_and_calculates_stable_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0001_create_example.sql"
            path.write_text("CREATE TABLE example (id integer);\n", encoding="utf-8")

            first = discover_migrations(Path(directory))
            second = discover_migrations(Path(directory))

        self.assertEqual([migration.version for migration in first], ["0001"])
        self.assertEqual(first[0].checksum, second[0].checksum)

    def test_rejects_unversioned_migration_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "create_example.sql"
            path.write_text("SELECT 1;\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Invalid migration filename"):
                discover_migrations(Path(directory))


if __name__ == "__main__":
    unittest.main()
