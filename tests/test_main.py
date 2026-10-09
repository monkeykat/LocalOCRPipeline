import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from app.database import DocumentStatus, ExtractionMethod, initialize_database
from app.extractor import EmbeddedPDF, ExtractionError, PDFExtraction
from app.main import BatchSummary, main, run_batch, scan_pdfs
from app.paths import get_project_paths
from app.processor import DocumentProcessor


class FakeExtractor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.attachments: dict[str, tuple[EmbeddedPDF, ...]] = {}
        self.fail_names: set[str] = set()

    def extract(self, source: Path | bytes, filename: str | None = None) -> PDFExtraction:
        name = filename or (source.name if isinstance(source, Path) else "embedded.pdf")
        self.calls.append(name)
        if name in self.fail_names:
            raise ExtractionError(f"cannot process {name}")
        return PDFExtraction(
            text=f"Text from {name}",
            page_count=3,
            extraction_method=ExtractionMethod.OCR,
            embedded_pdfs=self.attachments.get(name, ()),
        )


class MainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.paths = get_project_paths(Path(self.temp_dir.name))
        initialize_database(self.paths)
        self.extractor = FakeExtractor()
        self.processor = DocumentProcessor(self.paths, self.extractor)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def add_pdf(self, relative_path: str, content: bytes = b"%PDF-test") -> Path:
        path = self.paths.input_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_scan_returns_nested_pdf_paths_case_insensitively_sorted(self) -> None:
        first = self.add_pdf("a/first.PDF")
        second = self.add_pdf("second.pdf")
        self.add_pdf("nested/not-a-pdf.txt")
        self.add_pdf("nested/no-extension")

        self.assertEqual(scan_pdfs(self.paths.input_dir), [first, second])

    def test_scan_rejects_missing_input_directory(self) -> None:
        with self.assertRaises(FileNotFoundError):
            scan_pdfs(self.paths.input_dir / "missing")

    def test_batch_reports_totals_continues_after_failure_and_skips_duplicates(self) -> None:
        self.add_pdf("a/good.pdf", b"%PDF-good")
        self.add_pdf("b/bad.pdf", b"%PDF-bad")
        self.add_pdf("c/duplicate.pdf", b"%PDF-good")
        self.extractor.fail_names.add("bad.pdf")
        output = io.StringIO()

        summary = run_batch(
            self.paths,
            processor=self.processor,
            output=output,
        )
        report = output.getvalue()

        self.assertEqual(summary.discovered, 3)
        self.assertEqual(summary.documents, 1)
        self.assertEqual(summary.pages, 3)
        self.assertEqual(summary.native_pages, 0)
        self.assertEqual(summary.ocr_pages, 3)
        self.assertEqual(summary.failed, 1)
        self.assertEqual(self.extractor.calls.count("good.pdf"), 1)
        self.assertIn("Found: 3 PDFs", report)
        self.assertIn("ERROR: b/bad.pdf [bad.pdf]", report)
        self.assertIn("Reason: ExtractionError: cannot process bad.pdf", report)
        self.assertIn("Continuing...", report)
        self.assertIn("Pages:     3", report)
        self.assertIn("OCR:       3", report)
        self.assertIn("Failed:    1", report)
        self.assertIn("c/duplicate.pdf [duplicate.pdf]", report)
        self.assertIn("already complete; skipped", report)

    def test_batch_counts_recursive_embedded_documents(self) -> None:
        self.add_pdf("manual.pdf", b"%PDF-parent")
        self.extractor.attachments["manual.pdf"] = (
            EmbeddedPDF(
                filename="appendix.pdf",
                content=b"%PDF-child",
                parent_filename="manual.pdf",
                attachment_index=1,
            ),
        )
        output = io.StringIO()

        summary = run_batch(
            self.paths,
            processor=self.processor,
            output=output,
        )

        self.assertEqual(summary.discovered, 1)
        self.assertEqual(summary.documents, 2)
        self.assertEqual(summary.pages, 6)
        self.assertEqual(summary.ocr_pages, 6)
        self.assertIn("attachment", output.getvalue())

    def test_empty_input_batch_succeeds(self) -> None:
        output = io.StringIO()
        summary = run_batch(self.paths, processor=self.processor, output=output)

        self.assertEqual(summary.discovered, 0)
        self.assertEqual(summary.documents, 0)
        self.assertEqual(summary.failed, 0)
        self.assertIn("Found: 0 PDFs", output.getvalue())

    def test_main_returns_failure_status_for_batch_errors(self) -> None:
        with patch(
            "app.main.run_batch",
            return_value=BatchSummary(1, 0, 0, 0, 0, 1),
        ):
            self.assertEqual(main(), 1)

    def test_main_reports_setup_errors_to_stderr(self) -> None:
        stderr = io.StringIO()
        with (
            patch("app.main.run_batch", side_effect=FileNotFoundError("input missing")),
            redirect_stderr(stderr),
        ):
            self.assertEqual(main(), 1)
        self.assertIn("FileNotFoundError: input missing", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
