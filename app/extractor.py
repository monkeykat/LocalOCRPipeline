from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePath
from typing import Any, Protocol

import numpy as np

from app.database import ExtractionMethod

# Four pixels per PDF point is about 288 dpi, close to a common 300 dpi scan.
_OCR_IMAGE_SCALE = 4.0
_IMAGE_DOMINANT_PAGE_RATIO = 0.25
_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)?")
_SUSPICIOUS_WORD_PATTERN = re.compile(
    r"[A-Za-z][~`#$%&+<>;|][A-Za-z]|([A-Za-z])\1\1|"
    r"[A-Z][a-z]+[A-Z][a-z]"
)


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
    ocr_pages: tuple[int, ...] = ()
    ocr_confidence: tuple[tuple[int, float], ...] = ()


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
        image_scale: float = _OCR_IMAGE_SCALE,
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
        embedded_pdfs, image_pages = self._extract_pdf_resources(source, source_name)
        docling_image_pages, image_dominant_pages = self._get_docling_image_pages(
            document,
            page_items,
        )
        image_pages.update(docling_image_pages)
        pages_to_ocr = {
            page_number
            for page_number, page_text in native_pages.items()
            if self._native_text_is_suspicious(page_text)
        }
        pages_to_ocr.update(image_pages)
        pages_for_ocr = {
            page_number: page_items[page_number]
            for page_number in sorted(pages_to_ocr)
        }
        ocr_pages, ocr_confidence = self._extract_ocr_pages(
            pages_for_ocr,
            source_name,
        )

        page_texts: dict[int, str] = {}
        for page_number, native_text in native_pages.items():
            recognized_text = ocr_pages.get(page_number)
            if not recognized_text:
                page_texts[page_number] = native_text
            elif (
                page_number in image_pages
                and page_number not in image_dominant_pages
                and native_text.strip()
                and not self._native_text_is_suspicious(native_text)
            ):
                additional_ocr_text = self._ocr_lines_missing_from_native(
                    native_text,
                    recognized_text,
                )
                page_texts[page_number] = self._join_text(
                    native_text,
                    additional_ocr_text,
                )
            else:
                page_texts[page_number] = recognized_text

        method = (
            ExtractionMethod.OCR if pages_to_ocr else ExtractionMethod.NATIVE
        )
        return PDFExtraction(
            text=self._format_pages(page_texts),
            page_count=page_count,
            extraction_method=method,
            embedded_pdfs=embedded_pdfs,
            ocr_pages=tuple(sorted(ocr_pages)),
            ocr_confidence=tuple(sorted(ocr_confidence.items())),
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

    @staticmethod
    def _get_docling_image_pages(
        document: Any,
        pages: dict[int, Any],
    ) -> tuple[set[int], set[int]]:
        image_pages: set[int] = set()
        image_rectangles: dict[int, list[tuple[float, float, float, float]]] = {}
        image_dominant_pages: set[int] = set()
        for item, _ in document.iterate_items():
            label = getattr(item, "label", None)
            if getattr(label, "value", label) != "picture":
                continue

            provenance = getattr(item, "prov", ())
            if not provenance:
                return set(pages), set(pages)
            for source in provenance:
                page_number = int(source.page_no)
                if page_number in pages:
                    image_pages.add(page_number)
                    bbox = getattr(source, "bbox", None)
                    page_size = getattr(pages[page_number], "size", None)
                    try:
                        page_width = float(page_size.width)
                        page_height = float(page_size.height)
                        left = max(0.0, float(bbox.l))
                        right = min(page_width, float(bbox.r))
                        bottom = max(0.0, float(bbox.b))
                        top = min(page_height, float(bbox.t))
                    except (AttributeError, TypeError, ValueError):
                        image_dominant_pages.add(page_number)
                        continue
                    if (
                        page_width <= 0
                        or page_height <= 0
                        or right <= left
                        or top <= bottom
                    ):
                        image_dominant_pages.add(page_number)
                        continue
                    image_rectangles.setdefault(page_number, []).append(
                        (left, bottom, right, top)
                    )

        for page_number, rectangles in image_rectangles.items():
            page_size = pages[page_number].size
            page_area = float(page_size.width) * float(page_size.height)
            if page_area > 0 and (
                _rectangle_union_area(rectangles) / page_area
                >= _IMAGE_DOMINANT_PAGE_RATIO
            ):
                image_dominant_pages.add(page_number)
        return image_pages, image_dominant_pages

    @staticmethod
    def _native_text_is_suspicious(text: str) -> bool:
        if not text.strip():
            return True
        if "\ufffd" in text or any(
            ord(character) < 32 and character not in "\n\r\t"
            for character in text
        ):
            return True

        words = _WORD_PATTERN.findall(text)
        suspicious_words = sum(
            bool(_SUSPICIOUS_WORD_PATTERN.search(word)) for word in words
        )
        suspicious_characters = sum(
            character in "~`#$%&<>|^"
            for character in text
        )
        return suspicious_words >= 2 or suspicious_characters >= 2

    @staticmethod
    def _ocr_lines_missing_from_native(native_text: str, ocr_text: str) -> str:
        native_tokens = [
            token.casefold() for token in _WORD_PATTERN.findall(native_text)
        ]
        native_compact = "".join(
            character.casefold()
            for character in native_text
            if character.isalnum()
        )
        additional_lines: list[str] = []
        for line in ocr_text.splitlines():
            line = line.strip()
            if not line:
                continue
            line_tokens = [token.casefold() for token in _WORD_PATTERN.findall(line)]
            compact_line = "".join(
                character.casefold()
                for character in line
                if character.isalnum()
            )
            if not compact_line:
                continue
            if compact_line in native_compact or _contains_token_sequence(
                native_tokens,
                line_tokens,
            ):
                continue
            additional_lines.append(line)
        if not additional_lines:
            return ""
        return (
            "[OCR text not present in the PDF text layer]\n"
            + "\n".join(additional_lines)
        )

    @staticmethod
    def _join_text(first: str, second: str) -> str:
        if not first.strip():
            return second
        if not second.strip():
            return first
        return f"{first}\n\n{second}"

    def _extract_ocr_pages(
        self,
        pages: dict[int, Any],
        filename: str,
    ) -> tuple[dict[int, str], dict[int, float]]:
        if not pages:
            return {}, {}
        reader = self._get_ocr_reader()
        result: dict[int, str] = {}
        confidence: dict[int, float] = {}
        for page_number, page in pages.items():
            page_image = getattr(page, "image", None)
            pil_image = getattr(page_image, "pil_image", None)
            image = pil_image() if callable(pil_image) else pil_image
            if image is None:
                raise ExtractionError(
                    f"Docling did not provide a rendered image for page {page_number} "
                    f"of {filename}"
                )
            text, page_confidence = self._recognized_text(
                reader.ocr(np.asarray(image.convert("RGB")), cls=True),
                page_number,
            )
            result[page_number] = text
            if page_confidence is not None:
                confidence[page_number] = page_confidence
        return result, confidence

    def _get_ocr_reader(self) -> OCRReader:
        if self._ocr_reader is None:
            try:
                # PaddlePaddle 2.6 loads a bundled zlib symbol that can break
                # later imports of SciPy's native modules on ARM64.
                import scipy  # noqa: F401
                from paddleocr import PaddleOCR
            except ImportError as error:
                raise ExtractionError(
                    "PaddleOCR is required for scanned PDFs; install project requirements"
                ) from error
            self._ocr_reader = PaddleOCR(use_angle_cls=True, lang="en")
        return self._ocr_reader

    @staticmethod
    def _recognized_text(
        raw_result: Any,
        page_number: int,
    ) -> tuple[str, float | None]:
        if not isinstance(raw_result, list):
            raise ExtractionError(
                f"PaddleOCR returned an invalid result for page {page_number}"
            )
        if not raw_result or raw_result[0] is None:
            return "", None

        lines: list[str] = []
        confidence_scores: list[float] = []
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
                try:
                    confidence_scores.append(float(line[1][1]))
                except (IndexError, TypeError, ValueError):
                    pass
        mean_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else None
        )
        return "\n".join(lines), mean_confidence

    @staticmethod
    def _format_pages(pages: dict[int, str]) -> str:
        return "\n\n".join(
            f"=== PAGE {page_number} ===\n\n{text}"
            for page_number, text in pages.items()
        )

    @staticmethod
    def _extract_pdf_resources(
        source: Path | bytes,
        parent_filename: str,
    ) -> tuple[tuple[EmbeddedPDF, ...], set[int]]:
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

        reader = PdfReader(reader_source)
        image_pages: set[int] = set()
        for page_number, page in enumerate(getattr(reader, "pages", ()), start=1):
            try:
                images = page.images
                if images:
                    image_pages.add(page_number)
            except Exception:
                # An image the PDF reader cannot enumerate must not be silently
                # skipped. OCR the rendered page as a conservative fallback.
                image_pages.add(page_number)

        embedded: list[EmbeddedPDF] = []
        attachment_index = 0
        for name, contents_list in reader.attachments.items():
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
        return tuple(embedded), image_pages


def _contains_token_sequence(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    needle_length = len(needle)
    return any(
        haystack[index : index + needle_length] == needle
        for index in range(len(haystack) - needle_length + 1)
    )


def _rectangle_union_area(
    rectangles: list[tuple[float, float, float, float]],
) -> float:
    """Return the union area of page-coordinate rectangles."""
    x_coordinates = sorted(
        {
            coordinate
            for left, _, right, _ in rectangles
            for coordinate in (left, right)
        }
    )
    area = 0.0
    for left, right in zip(x_coordinates, x_coordinates[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (bottom, top)
            for rect_left, bottom, rect_right, top in rectangles
            if rect_left < right and rect_right > left
        )
        covered_height = 0.0
        if intervals:
            interval_start, interval_end = intervals[0]
            for start, end in intervals[1:]:
                if start <= interval_end:
                    interval_end = max(interval_end, end)
                else:
                    covered_height += interval_end - interval_start
                    interval_start, interval_end = start, end
            covered_height += interval_end - interval_start
        area += (right - left) * covered_height
    return area


def _is_pdf_attachment(name: str, content: bytes) -> bool:
    return name.lower().endswith(".pdf") or b"%PDF-" in content[:1024]


def _safe_pdf_filename(name: str, attachment_index: int) -> str:
    filename = PurePath(name.replace("\\", "/")).name
    if not filename:
        filename = f"embedded-{attachment_index}.pdf"
    elif not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"
    return filename
