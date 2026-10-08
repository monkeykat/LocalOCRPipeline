from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePath
from typing import Any, Protocol

import numpy as np

from app.database import ExtractionMethod


class ExtractionError(RuntimeError):
    """Raised when a PDF cannot be converted into page-preserved text."""


@dataclass(frozen=True)
class EmbeddedPDF:
    filename: str
    content: bytes
    parent_filename: str
    attachment_index: int


@dataclass(frozen=True)
class PDFExtraction:
    text: str
    page_count: int
    extraction_method: ExtractionMethod
    embedded_pdfs: tuple[EmbeddedPDF, ...]


class OCRReader(Protocol):
    def ocr(self, image: np.ndarray, cls: bool = ...) -> Any: ...


class PDFConverter(Protocol):
    def convert(self, source: Any) -> Any: ...


class PdfExtractor:
    def __init__(
        self,
        converter: PDFConverter | None = None,
        ocr_reader: OCRReader | None = None,
        *,
        image_scale: float = 2.0,
    ) -> None:
        if image_scale <= 0:
            raise ValueError("image_scale must be greater than zero")
        self._converter = converter
        self._ocr_reader = ocr_reader
        self._image_scale = image_scale

    def extract(self, source: Path | bytes, filename: str | None = None) -> PDFExtraction:
        source_name = filename or (source.name if isinstance(source, Path) else None)
        if not source_name:
            raise ValueError("filename is required when extracting PDF bytes")

        conversion_source = self._conversion_source(source, source_name)
        conversion_result = self._get_converter().convert(conversion_source)
        self._validate_conversion_result(conversion_result, source_name)
        document = conversion_result.document
        page_items = self._get_page_items(document)
        page_count = len(page_items)
        if page_count == 0:
            raise ExtractionError(f"Docling found no pages in {source_name}")
        expected_page_count = getattr(
            getattr(conversion_result, "input", None),
            "page_count",
            page_count,
        )
        if expected_page_count != page_count:
            raise ExtractionError(
                f"Docling returned {page_count} page records for {source_name}; "
                f"the PDF has {expected_page_count} pages"
            )

        native_pages = self._extract_native_pages(document, page_items)
        if all(page_text.strip() for page_text in native_pages.values()):
            page_texts = native_pages
            method = ExtractionMethod.NATIVE
        else:
            page_texts = self._extract_ocr_pages(page_items, source_name)
            method = ExtractionMethod.OCR

        embedded_pdfs = self._extract_embedded_pdfs(source, source_name)
        return PDFExtraction(
            text=self._format_pages(page_texts),
            page_count=page_count,
            extraction_method=method,
            embedded_pdfs=embedded_pdfs,
        )

    def _get_converter(self) -> PDFConverter:
        if self._converter is None:
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import NativePdfPipelineOptions
                from docling.document_converter import (
                    DocumentConverter,
                    NativePdfFormatOption,
                )
            except ImportError as error:
                raise ExtractionError(
                    "Docling is required for PDF parsing; install project requirements"
                ) from error

            pipeline_options = NativePdfPipelineOptions(
                generate_page_images=True,
                images_scale=self._image_scale,
            )
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: NativePdfFormatOption(
                        pipeline_options=pipeline_options,
                    )
                }
            )
        return self._converter

    @staticmethod
    def _conversion_source(source: Path | bytes, filename: str) -> Any:
        if isinstance(source, Path):
            return source

        try:
            from docling.datamodel.base_models import DocumentStream
        except ImportError as error:
            raise ExtractionError(
                "Docling is required for PDF parsing; install project requirements"
            ) from error
        return DocumentStream(name=filename, stream=BytesIO(source))

    @staticmethod
    def _get_page_items(document: Any) -> dict[int, Any]:
        pages: dict[int, Any] = {}
        for page in document.pages.values():
            page_number = int(page.page_no)
            if page_number < 1:
                raise ExtractionError(
                    f"Docling returned invalid page number {page_number}"
                )
            if page_number in pages:
                raise ExtractionError(f"Docling returned duplicate page {page_number}")
            pages[page_number] = page
        ordered_pages = dict(sorted(pages.items()))
        expected_numbers = list(range(1, len(ordered_pages) + 1))
        if list(ordered_pages) != expected_numbers:
            raise ExtractionError("Docling returned non-contiguous PDF page numbers")
        return ordered_pages

    @staticmethod
    def _validate_conversion_result(result: Any, filename: str) -> None:
        status = getattr(result, "status", None)
        status_name = getattr(status, "name", None)
        if status_name is None:
            raise ExtractionError(
                f"Docling conversion of {filename} did not report a status"
            )
        if status_name == "SUCCESS":
            return
        error_messages = [
            str(getattr(error, "error_message", error))
            for error in getattr(result, "errors", ())
        ]
        details = "; ".join(error_messages) or "no error details were provided"
        raise ExtractionError(
            f"Docling conversion of {filename} ended with {status_name}: {details}"
        )

    @staticmethod
    def _extract_native_pages(document: Any, pages: dict[int, Any]) -> dict[int, str]:
        page_text: dict[int, list[str]] = {page_number: [] for page_number in pages}
        for item, _ in document.iterate_items():
            provenance = getattr(item, "prov", ())
            text = getattr(item, "text", None)
            if not isinstance(text, str):
                if getattr(item, "label", None) == "table":
                    text = item.export_to_markdown(doc=document)
                else:
                    continue

            for source in provenance:
                page_number = int(source.page_no)
                if page_number in page_text and text.strip():
                    page_text[page_number].append(text.strip())

        return {
            page_number: "\n".join(items)
            for page_number, items in page_text.items()
        }

    def _extract_ocr_pages(
        self,
        pages: dict[int, Any],
        filename: str,
    ) -> dict[int, str]:
        reader = self._get_ocr_reader()
        result: dict[int, str] = {}
        for page_number, page in pages.items():
            page_image = getattr(page, "image", None)
            image_factory = getattr(page_image, "pil_image", None)
            image = image_factory() if callable(image_factory) else None
            if image is None:
                raise ExtractionError(
                    f"Docling did not provide a rendered image for page {page_number} "
                    f"of {filename}"
                )
            result[page_number] = self._recognized_text(
                reader.ocr(np.asarray(image.convert("RGB")), cls=True),
                page_number,
            )
        return result

    def _get_ocr_reader(self) -> OCRReader:
        if self._ocr_reader is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as error:
                raise ExtractionError(
                    "PaddleOCR is required for scanned PDFs; install project requirements"
                ) from error
            self._ocr_reader = PaddleOCR(use_angle_cls=True, lang="en")
        return self._ocr_reader

    @staticmethod
    def _recognized_text(raw_result: Any, page_number: int) -> str:
        if not isinstance(raw_result, list):
            raise ExtractionError(
                f"PaddleOCR returned an invalid result for page {page_number}"
            )
        if not raw_result or raw_result[0] is None:
            return ""

        lines: list[str] = []
        for line in raw_result[0]:
            try:
                recognized_text = line[1][0]
            except (IndexError, TypeError) as error:
                raise ExtractionError(
                    f"PaddleOCR returned a malformed line for page {page_number}"
                ) from error
            if not isinstance(recognized_text, str):
                raise ExtractionError(
                    f"PaddleOCR returned non-text content for page {page_number}"
                )
            if recognized_text.strip():
                lines.append(recognized_text.strip())
        return "\n".join(lines)

    @staticmethod
    def _format_pages(pages: dict[int, str]) -> str:
        return "\n\n".join(
            f"=== PAGE {page_number} ===\n\n{text}"
            for page_number, text in pages.items()
        )

    @staticmethod
    def _extract_embedded_pdfs(
        source: Path | bytes,
        parent_filename: str,
    ) -> tuple[EmbeddedPDF, ...]:
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise ExtractionError(
                "pypdf is required to extract embedded PDF attachments"
            ) from error

        reader_source: Path | BytesIO
        if isinstance(source, Path):
            reader_source = source
        else:
            reader_source = BytesIO(source)

        embedded: list[EmbeddedPDF] = []
        attachment_index = 0
        for name, contents_list in PdfReader(reader_source).attachments.items():
            for contents in contents_list:
                attachment_index += 1
                if not _is_pdf_attachment(name, contents):
                    continue
                embedded.append(
                    EmbeddedPDF(
                        filename=_safe_pdf_filename(name, attachment_index),
                        content=contents,
                        parent_filename=parent_filename,
                        attachment_index=attachment_index,
                    )
                )
        return tuple(embedded)


def _is_pdf_attachment(name: str, content: bytes) -> bool:
    return name.lower().endswith(".pdf") or b"%PDF-" in content[:1024]


def _safe_pdf_filename(name: str, attachment_index: int) -> str:
    filename = PurePath(name.replace("\\", "/")).name
    if not filename:
        filename = f"embedded-{attachment_index}.pdf"
    elif not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"
    return filename
