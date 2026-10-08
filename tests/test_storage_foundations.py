import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.database import (
    DocumentStatus,
    ExtractionMethod,
    complete_document,
    connect_database,
    has_completed_hash,
    initialize_database,
    record_error,
    register_document,
    set_document_status,
    update_page_count,
)
from app.paths import (
    ensure_data_directories,
    failed_path_for,
    get_project_paths,
    output_path_for,
)


class DatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.paths = get_project_paths(Path(self.temp_dir.name))
        initialize_database(self.paths)
        self.connection = connect_database(self.paths.database_path)

    def tearDown(self) -> None:
        self.connection.close()
        self.temp_dir.cleanup()

    def test_register_document_is_idempotent_for_path_and_hash(self) -> None:
        first_id = register_document(
            self.connection,
            filename="manual.pdf",
            file_path="/input/manual.pdf",
            file_hash="hash-1",
        )
        second_id = register_document(
            self.connection,
            filename="manual.pdf",
            file_path="/input/manual.pdf",
            file_hash="hash-1",
        )

        self.assertEqual(first_id, second_id)
        row = self.connection.execute(
            "SELECT filename, file_path, file_hash, status, created_at "
            "FROM documents WHERE id = ?",
            (first_id,),
        ).fetchone()
        self.assertEqual(row["filename"], "manual.pdf")
        self.assertEqual(row["file_path"], "/input/manual.pdf")
        self.assertEqual(row["file_hash"], "hash-1")
        self.assertEqual(row["status"], DocumentStatus.PENDING.value)
        self.assertTrue(row["created_at"])

    def test_schema_has_architecture_fields_and_error_foreign_key(self) -> None:
        document_columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(documents)")
        }
        error_columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(errors)")
        }
        error_foreign_keys = list(
            self.connection.execute("PRAGMA foreign_key_list(errors)")
        )

        self.assertEqual(
            document_columns,
            {
                "id",
                "filename",
                "file_path",
                "file_hash",
                "page_count",
                "extraction_method",
                "status",
                "created_at",
                "completed_at",
            },
        )
        self.assertEqual(
            error_columns,
            {"id", "document_id", "error", "created_at"},
        )
        self.assertEqual(error_foreign_keys[0]["table"], "documents")
        self.assertEqual(error_foreign_keys[0]["from"], "document_id")

    def test_completion_tracking_and_error_foreign_key(self) -> None:
        document_id = register_document(
            self.connection,
            filename="scan.pdf",
            file_path="/input/scan.pdf",
            file_hash="scan-hash",
        )
        set_document_status(self.connection, document_id, DocumentStatus.PROCESSING)
        update_page_count(self.connection, document_id, 3)

        self.assertFalse(has_completed_hash(self.connection, "scan-hash"))

        error_id = record_error(self.connection, document_id, "Unreadable page")
        error_row = self.connection.execute(
            "SELECT document_id, error, created_at FROM errors WHERE id = ?",
            (error_id,),
        ).fetchone()
        self.assertEqual(error_row["document_id"], document_id)
        self.assertEqual(error_row["error"], "Unreadable page")
        self.assertTrue(error_row["created_at"])

        complete_document(
            self.connection,
            document_id,
            page_count=3,
            extraction_method=ExtractionMethod.OCR,
        )
        self.assertTrue(has_completed_hash(self.connection, "scan-hash"))
        completed_row = self.connection.execute(
            "SELECT status, page_count, extraction_method, completed_at "
            "FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        self.assertEqual(completed_row["status"], DocumentStatus.COMPLETE.value)
        self.assertEqual(completed_row["page_count"], 3)
        self.assertEqual(completed_row["extraction_method"], "ocr")
        self.assertTrue(completed_row["completed_at"])

    def test_invalid_status_and_orphan_error_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            set_document_status(
                self.connection,
                1,
                DocumentStatus.COMPLETE,
            )

        document_id = register_document(
            self.connection,
            filename="manual.pdf",
            file_path="/input/manual.pdf",
            file_hash="manual-hash",
        )
        with self.assertRaises(ValueError):
            complete_document(
                self.connection,
                document_id,
                page_count=1,
                extraction_method=ExtractionMethod.NATIVE,
            )

        with self.assertRaises(sqlite3.IntegrityError):
            record_error(self.connection, 999, "Orphan error")


class ProjectPathsTests(unittest.TestCase):
    def test_directories_and_nested_source_paths_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = get_project_paths(Path(temp_dir))
            ensure_data_directories(paths)
            source = paths.input_dir / "manuals" / "systems" / "engine.pdf"

            self.assertTrue(paths.input_dir.is_dir())
            self.assertTrue(paths.output_dir.is_dir())
            self.assertTrue(paths.failed_dir.is_dir())
            self.assertEqual(
                output_path_for(source, paths),
                paths.output_dir / "manuals" / "systems" / "engine.txt",
            )
            self.assertEqual(
                failed_path_for(source, paths),
                paths.failed_dir / "manuals" / "systems" / "engine.pdf",
            )

    def test_paths_outside_input_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = get_project_paths(Path(temp_dir))
            with self.assertRaises(ValueError):
                output_path_for(Path(temp_dir) / "other.pdf", paths)


if __name__ == "__main__":
    unittest.main()
