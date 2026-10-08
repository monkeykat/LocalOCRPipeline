import sys
import types
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pypdf

from app.database import ExtractionMethod
from app.extractor import ExtractionError, PdfExtractor


class FakeImage:
    def convert(self, mode: str) -> list[list[int]]:
        if mode != "RGB":
            raise AssertionError(f"Unexpected image mode: {mode}")
        return [[1, 2], [3, 4]]


class FakeOCRReader:
    def __init__(self, responses: list[list[tuple[object, tuple[str, float]]]]) -> None:
        self.responses = responses
        self.images: list[object] = []

    def ocr(self, image: object, cls: bool = False) -> object:
        self.images.append(image)
        return [self.responses[len(self.images) - 1]]


class FakeConverter:
    def __init__(
        self,
        document: object,
        *,
        status: object | None = None,
        page_count: int | None = None,
        errors: list[object] | None = None,
    ) -> None:
        self.document = document
        self.status = status or SimpleNamespace(name="SUCCESS")
        self.page_count = page_count
        self.errors = errors or []
        self.sources: list[object] = []

    def convert(self, source: object) -> object:
        self.sources.append(source)
        conversion = SimpleNamespace(
            document=self.document,
            errors=self.errors,
        )
        if self.status is not None:
            conversion.status = self.status
        if self.page_count is not None:
            conversion.input = SimpleNamespace(page_count=self.page_count)
        return conversion


def make_document(
    pages: dict[int, object],
    items: list[tuple[object, int]],
) -> SimpleNamespace:
    return SimpleNamespace(
        pages=pages,
        iterate_items=lambda: iter(items),
    )


def make_page(page_number: int, image: object | None = None) -> SimpleNamespace:
    return SimpleNamespace(page_no=page_number, image=image)


def make_text_item(text: str, page_number: int) -> SimpleNamespace:
    return SimpleNamespace(text=text, prov=[SimpleNamespace(page_no=page_number)])


class PdfExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        fake_reader_module = types.ModuleType("pypdf")

        class EmptyPdfReader:
            def __init__(self, source: object) -> None:
                self.attachments: dict[str, list[bytes]] = {}

        fake_reader_module.PdfReader = EmptyPdfReader
        patcher = patch.dict(sys.modules, {"pypdf": fake_reader_module})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_uses_native_text_when_all_pages_have_usable_text(self) -> None:
        document = make_document(
            {1: make_page(1), 2: make_page(2)},
            [(make_text_item("First page", 1), 0), (make_text_item("Second page", 2), 0)],
        )
        converter = FakeConverter(document)

        result = PdfExtractor(converter=converter).extract(
            Path("/input/manual.pdf")
        )

        self.assertEqual(result.extraction_method, ExtractionMethod.NATIVE)
        self.assertEqual(result.page_count, 2)
        self.assertEqual(
            result.text,
            "=== PAGE 1 ===\n\nFirst page\n\n=== PAGE 2 ===\n\nSecond page",
        )
        self.assertEqual(converter.sources, [Path("/input/manual.pdf")])

    def test_document_without_usable_text_uses_ocr_for_every_page(self) -> None:
        pages = {
            1: make_page(1, SimpleNamespace(pil_image=lambda: FakeImage())),
            2: make_page(2, SimpleNamespace(pil_image=lambda: FakeImage())),
        }
        document = make_document(
            pages,
            [(make_text_item("Native on page one", 1), 0)],
        )
        ocr = FakeOCRReader(
            [
                [(object(), ("OCR page one", 0.99))],
                [],
            ]
        )
        result = PdfExtractor(
            converter=FakeConverter(document),
            ocr_reader=ocr,
        ).extract(Path("/input/scanned.pdf"))

        self.assertEqual(result.extraction_method, ExtractionMethod.OCR)
        self.assertEqual(result.page_count, 2)
        self.assertEqual(len(ocr.images), 2)
        self.assertEqual(
            result.text,
            "=== PAGE 1 ===\n\nOCR page one\n\n=== PAGE 2 ===\n\n",
        )

    def test_table_content_is_exported_to_markdown(self) -> None:
        table = SimpleNamespace(
            label="table",
            export_to_markdown=lambda doc: "| Part | Value |\n| --- | --- |\n| A | 2 |",
            prov=[SimpleNamespace(page_no=1)],
        )
        document = make_document({1: make_page(1)}, [(table, 0)])

        result = PdfExtractor(converter=FakeConverter(document)).extract(
            Path("/input/table.pdf")
        )

        self.assertEqual(result.extraction_method, ExtractionMethod.NATIVE)
        self.assertIn("| Part | Value |", result.text)

    def test_empty_docling_page_map_is_an_extraction_error(self) -> None:
        document = make_document({}, [])
        with self.assertRaisesRegex(ExtractionError, "found no pages"):
            PdfExtractor(converter=FakeConverter(document)).extract(
                Path("/input/empty.pdf")
            )

    def test_page_count_mismatch_is_not_silently_accepted(self) -> None:
        document = make_document({1: make_page(1)}, [(make_text_item("Page", 1), 0)])
        with self.assertRaisesRegex(ExtractionError, "the PDF has 2 pages"):
            PdfExtractor(
                converter=FakeConverter(document, page_count=2)
            ).extract(Path("/input/truncated.pdf"))

    def test_non_contiguous_page_numbers_are_not_silently_accepted(self) -> None:
        document = make_document(
            {1: make_page(1), 3: make_page(3)},
            [(make_text_item("Page one", 1), 0), (make_text_item("Page three", 3), 0)],
        )
        with self.assertRaisesRegex(ExtractionError, "non-contiguous"):
            PdfExtractor(converter=FakeConverter(document)).extract(
                Path("/input/gapped.pdf")
            )

    def test_failed_docling_conversion_is_reported(self) -> None:
        converter = FakeConverter(
            make_document({}, []),
            status=SimpleNamespace(name="FAILURE"),
            errors=[SimpleNamespace(error_message="damaged PDF")],
        )
        with self.assertRaisesRegex(ExtractionError, "damaged PDF"):
            PdfExtractor(converter=converter).extract(Path("/input/damaged.pdf"))

    def test_missing_page_image_is_reported_during_ocr(self) -> None:
        document = make_document({1: make_page(1)}, [])
        with self.assertRaisesRegex(ExtractionError, "did not provide a rendered image"):
            PdfExtractor(
                converter=FakeConverter(document),
                ocr_reader=FakeOCRReader([]),
            ).extract(Path("/input/scanned.pdf"))

    def test_malformed_ocr_result_is_reported(self) -> None:
        with self.assertRaisesRegex(ExtractionError, "malformed line"):
            PdfExtractor._recognized_text([[(object(),)]], 4)

    def test_embedded_pdf_attachments_are_returned_with_source_context(self) -> None:
        fake_reader_module = types.ModuleType("pypdf")

        class FakePdfReader:
            def __init__(self, source: object) -> None:
                self.source = source
                self.attachments = {
                    "../manual.pdf": [b"%PDF-1.7 embedded"],
                    "notes.txt": [b"not a PDF"],
                    "unnamed": [b"%PDF-1.7 embedded without extension"],
                }

        fake_reader_module.PdfReader = FakePdfReader
        with patch.dict(sys.modules, {"pypdf": fake_reader_module}):
            result = PdfExtractor._extract_embedded_pdfs(
                Path("/input/parent.pdf"),
                "parent.pdf",
            )

        self.assertEqual(
            [(item.filename, item.attachment_index) for item in result],
            [("manual.pdf", 1), ("unnamed.pdf", 3)],
        )
        self.assertEqual(result[0].parent_filename, "parent.pdf")
        self.assertEqual(result[0].content, b"%PDF-1.7 embedded")

    def test_embedded_pdf_duplicate_names_keep_distinct_attachment_indices(self) -> None:
        fake_reader_module = types.ModuleType("pypdf")

        class FakePdfReader:
            def __init__(self, source: object) -> None:
                self.attachments = {
                    "manual.pdf": [b"%PDF-first", b"%PDF-second"],
                }

        fake_reader_module.PdfReader = FakePdfReader
        with patch.dict(sys.modules, {"pypdf": fake_reader_module}):
            result = PdfExtractor._extract_embedded_pdfs(
                b"%PDF-parent",
                "parent.pdf",
            )

        self.assertEqual([item.attachment_index for item in result], [1, 2])
        self.assertNotEqual(result[0].content, result[1].content)

    def test_pypdf_extracts_real_embedded_pdf_attachment(self) -> None:
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.add_attachment("manual.pdf", b"%PDF-1.7 child")
        parent_pdf = BytesIO()
        writer.write(parent_pdf)

        with patch.dict(sys.modules, {"pypdf": pypdf}):
            result = PdfExtractor._extract_embedded_pdfs(
                parent_pdf.getvalue(),
                "parent.pdf",
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].filename, "manual.pdf")
        self.assertEqual(result[0].content, b"%PDF-1.7 child")


if __name__ == "__main__":
    unittest.main()
