import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.database import (
    DocumentStatus,
    ExtractionMethod,
    connect_database,
    initialize_database,
    register_document,
    set_document_status,
)
from app.extractor import EmbeddedPDF, ExtractionError, PDFExtraction
from app.paths import get_project_paths
from app.processor import DocumentProcessor


class FakeExtractor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.attachments: dict[str, tuple[EmbeddedPDF, ...]] = {}
        self.fail_names: set[str] = set()

    def extract(self, source: Path | bytes, filename: str | None = None) -> PDFExtraction:
        name = filename or (source.name if isinstance(source, Path) else "unknown.pdf")
        self.calls.append(name)
        if name in self.fail_names:
            raise ExtractionError(f"cannot read {name}")
        return PDFExtraction(
            text=f"Text from {name}",
            page_count=2,
            extraction_method=ExtractionMethod.NATIVE,
            embedded_pdfs=self.attachments.get(name, ()),
        )


class DocumentProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.paths = get_project_paths(self.root)
        initialize_database(self.paths)
        self.extractor = FakeExtractor()
        self.processor = DocumentProcessor(self.paths, self.extractor)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def create_pdf(self, relative_path: str, content: bytes = b"%PDF-1.7 source") -> Path:
        source = self.paths.input_dir / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)
        return source

    def database_rows(self) -> list[object]:
        with connect_database(self.paths.database_path) as connection:
            return list(
                connection.execute(
                    "SELECT * FROM documents ORDER BY id"
                ).fetchall()
            )

    def test_success_publishes_text_and_metadata_then_marks_complete(self) -> None:
        source = self.create_pdf("manuals/engine.pdf")

        result = self.processor.process_document(source)

        output_path = self.paths.output_dir / "manuals" / "engine.txt"
        metadata_path = output_path.with_suffix(".json")
        self.assertEqual(result.status, DocumentStatus.COMPLETE)
        self.assertEqual(result.page_count, 2)
        self.assertEqual(result.extraction_method, ExtractionMethod.NATIVE)
        self.assertEqual(
            output_path.read_text(encoding="utf-8"),
            "DOCUMENT: engine.pdf\n\nText from engine.pdf\n",
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["filename"], "engine.pdf")
        self.assertEqual(metadata["page_count"], 2)
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(self.database_rows()[0]["status"], "COMPLETE")
        self.assertEqual(source.read_bytes(), b"%PDF-1.7 source")

    def test_completed_hash_is_not_extracted_twice(self) -> None:
        original = self.create_pdf("one/original.pdf", b"%PDF-same")
        copy = self.create_pdf("two/copy.pdf", b"%PDF-same")

        self.processor.process_document(original)
        result = self.processor.process_document(copy)

        self.assertTrue(result.skipped)
        self.assertEqual(self.extractor.calls, ["original.pdf"])
        self.assertEqual(len(self.database_rows()), 2)
        copied_output = self.paths.output_dir / "two" / "copy.txt"
        self.assertTrue(copied_output.is_file())
        self.assertEqual(
            copied_output.read_text(encoding="utf-8"),
            "DOCUMENT: copy.pdf\n\nText from original.pdf\n",
        )

    def test_missing_cached_output_is_recreated_from_source(self) -> None:
        source = self.create_pdf("missing-output.pdf", b"%PDF-source")
        self.processor.process_document(source)
        (self.paths.output_dir / "missing-output.txt").unlink()
        (self.paths.output_dir / "missing-output.json").unlink()

        result = self.processor.process_document(source)

        self.assertEqual(result.status, DocumentStatus.COMPLETE)
        self.assertFalse(result.skipped)
        self.assertEqual(self.extractor.calls, ["missing-output.pdf"] * 2)
        self.assertTrue((self.paths.output_dir / "missing-output.txt").is_file())

    def test_failure_records_error_and_copies_source_without_stopping(self) -> None:
        first = self.create_pdf("damaged.pdf", b"%PDF-damaged")
        second = self.create_pdf("good.pdf", b"%PDF-good")
        self.extractor.fail_names.add("damaged.pdf")

        failure = self.processor.process_document(first)
        success = self.processor.process_document(second)

        self.assertEqual(failure.status, DocumentStatus.FAILED)
        self.assertIn("cannot read damaged.pdf", failure.error or "")
        self.assertTrue((self.paths.failed_dir / "damaged.pdf").is_file())
        self.assertEqual(success.status, DocumentStatus.COMPLETE)
        self.assertEqual([row["status"] for row in self.database_rows()], ["FAILED", "COMPLETE"])
        with connect_database(self.paths.database_path) as connection:
            error_row = connection.execute(
                "SELECT error FROM errors WHERE document_id = ?",
                (self.database_rows()[0]["id"],),
            ).fetchone()
        self.assertIn("cannot read damaged.pdf", error_row["error"])

    def test_output_write_failure_never_leaves_a_success_shaped_result(self) -> None:
        source = self.create_pdf("write-fails.pdf", b"%PDF-write-fails")

        def fail_text_write(destination: Path, text: str) -> None:
            if destination.suffix == ".txt":
                raise OSError("disk full")
            destination.write_text(text, encoding="utf-8")

        with patch("app.processor._atomic_write_text", side_effect=fail_text_write):
            result = self.processor.process_document(source)

        output_path = self.paths.output_dir / "write-fails.txt"
        self.assertEqual(result.status, DocumentStatus.FAILED)
        self.assertFalse(output_path.exists())
        self.assertFalse(output_path.with_suffix(".json").exists())
        self.assertEqual(self.database_rows()[0]["status"], "FAILED")

    def test_changed_source_failure_removes_stale_output(self) -> None:
        source = self.create_pdf("changed.pdf", b"%PDF-first")
        self.processor.process_document(source)
        source.write_bytes(b"%PDF-second")
        self.extractor.fail_names.add("changed.pdf")

        result = self.processor.process_document(source)

        output_path = self.paths.output_dir / "changed.txt"
        self.assertEqual(result.status, DocumentStatus.FAILED)
        self.assertFalse(output_path.exists())
        self.assertFalse(output_path.with_suffix(".json").exists())
        self.assertEqual(source.read_bytes(), b"%PDF-second")

    def test_interrupted_processing_row_is_reset_to_pending_on_startup(self) -> None:
        source = self.create_pdf("interrupted.pdf")
        with connect_database(self.paths.database_path) as connection:
            document_id = register_document(
                connection,
                filename=source.name,
                file_path=str(source.resolve()),
                file_hash="interrupted-hash",
            )
            set_document_status(
                connection,
                document_id,
                DocumentStatus.PROCESSING,
            )

        restarted = DocumentProcessor(self.paths, self.extractor)

        self.assertEqual(self.database_rows()[0]["status"], "PENDING")
        self.assertEqual(restarted.process_document(source).status, DocumentStatus.COMPLETE)

    def test_embedded_pdfs_are_processed_recursively_with_unique_outputs(self) -> None:
        source = self.create_pdf("parent.pdf", b"%PDF-parent")
        child_one = EmbeddedPDF(
            filename="manual.pdf",
            content=b"%PDF-child-one",
            parent_filename="parent.pdf",
            attachment_index=1,
        )
        child_two = EmbeddedPDF(
            filename="manual.pdf",
            content=b"%PDF-child-two",
            parent_filename="parent.pdf",
            attachment_index=2,
        )
        grandchild = EmbeddedPDF(
            filename="appendix.pdf",
            content=b"%PDF-grandchild",
            parent_filename="manual.pdf",
            attachment_index=1,
        )
        self.extractor.attachments["parent.pdf"] = (child_one, child_two)
        self.extractor.attachments["manual.pdf"] = (grandchild,)

        result = self.processor.process_document(source)

        self.assertEqual(result.status, DocumentStatus.COMPLETE)
        self.assertEqual(len(result.children), 2)
        self.assertEqual(result.children[0].children[0].filename, "appendix.pdf")
        child_path_one = (
            self.paths.output_dir / "parent.embedded" / "0001-manual.txt"
        )
        child_path_two = (
            self.paths.output_dir / "parent.embedded" / "0002-manual.txt"
        )
        nested_path = (
            self.paths.output_dir
            / "parent.embedded"
            / "0001-manual.embedded"
            / "0001-appendix.txt"
        )
        second_nested_path = (
            self.paths.output_dir
            / "parent.embedded"
            / "0002-manual.embedded"
            / "0001-appendix.txt"
        )
        self.assertTrue(child_path_one.is_file())
        self.assertTrue(child_path_two.is_file())
        self.assertTrue(nested_path.is_file())
        self.assertTrue(second_nested_path.is_file())
        self.assertEqual(len(self.database_rows()), 5)
        child_metadata = json.loads(child_path_one.with_suffix(".json").read_text())
        self.assertEqual(child_metadata["parent"]["filename"], "parent.pdf")
        self.assertEqual(child_metadata["parent"]["attachment_index"], 1)

    def test_embedded_content_cycle_is_recorded_as_failure(self) -> None:
        source = self.create_pdf("cycle.pdf", b"%PDF-cycle")
        self.extractor.attachments["cycle.pdf"] = (
            EmbeddedPDF(
                filename="cycle-copy.pdf",
                content=b"%PDF-cycle",
                parent_filename="cycle.pdf",
                attachment_index=1,
            ),
        )

        result = self.processor.process_document(source)

        self.assertEqual(result.status, DocumentStatus.COMPLETE)
        self.assertEqual(result.children[0].status, DocumentStatus.FAILED)
        self.assertIn("cycle detected", result.children[0].error or "")
        self.assertEqual(
            len(list((self.paths.failed_dir / "embedded").rglob("0001-cycle-copy.pdf"))),
            1,
        )


if __name__ == "__main__":
    unittest.main()
