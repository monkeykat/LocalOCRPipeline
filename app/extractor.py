from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from io import BytesIO
from math import isfinite
from pathlib import Path, PurePath
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from app.database import ExtractionMethod
from app.text_quality import assess_native_text

# Four pixels per PDF point is about 288 dpi, close to a common 300 dpi scan.
_OCR_IMAGE_SCALE = 4.0
_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)?")


class ExtractionError(RuntimeError):
    """Raised when a PDF cannot be converted into page-preserved text."""


@dataclass(frozen=True)
class EmbeddedPDF:
    filename: str
    content: bytes
    parent_filename: str
    attachment_index: int


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float | None
    polygon: tuple[tuple[float, float], ...] | None


@dataclass(frozen=True)
class NativeTextRegion:
    text: str
    bbox: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class PDFExtraction:
    text: str
    page_count: int
    extraction_method: ExtractionMethod
    embedded_pdfs: tuple[EmbeddedPDF, ...]
    ocr_pages: tuple[int, ...] = ()
    ocr_confidence: tuple[tuple[int, float], ...] = ()
    ocr_lines: tuple[tuple[int, tuple[OCRLine, ...]], ...] = ()
    native_regions: tuple[tuple[int, tuple[NativeTextRegion, ...]], ...] = ()
    ocr_engine: str = "paddleocr"
    ocr_time_seconds: float | None = None
    extraction_time_seconds: float | None = None
    quality_assessment: dict[str, Any] | None = None
    fusion_assessment: dict[str, Any] | None = None
    native_text_assessment: dict[str, Any] | None = None


class OCRReader(Protocol):
    def ocr(self, image: np.ndarray, cls: bool = ...) -> Any: ...


class TesseractOCRReader:
    """Adapt Tesseract's word-level TSV output to the pipeline's OCR format."""

    def __init__(self, pytesseract: Any) -> None:
        self._pytesseract = pytesseract

    def ocr(self, image: np.ndarray, cls: bool = False) -> list[list[Any]]:
        del cls  # PaddleOCR's compatibility argument has no Tesseract equivalent.
        data = self._pytesseract.image_to_data(
            image,
            lang="eng",
            config="--oem 1 --psm 1",
            output_type=self._pytesseract.Output.DICT,
        )
        grouped_words: dict[tuple[int, int, int, int], list[tuple[str, float, int, int, int, int]]] = {}
        count = len(data.get("text", ()))
        for index in range(count):
            text = str(data["text"][index]).strip()
            if not text:
                continue
            try:
                confidence = float(data["conf"][index])
                left = int(data["left"][index])
                top = int(data["top"][index])
                width = int(data["width"][index])
                height = int(data["height"][index])
                key = tuple(
                    int(data.get(field, [0] * count)[index])
                    for field in ("page_num", "block_num", "par_num", "line_num")
                )
            except (IndexError, TypeError, ValueError):
                continue
            if confidence < 0 or width <= 0 or height <= 0:
                continue
            grouped_words.setdefault(key, []).append(
                (text, confidence, left, top, width, height)
            )

        lines: list[Any] = []
        for words in grouped_words.values():
            line_text = " ".join(word[0] for word in words)
            line_confidence = sum(word[1] for word in words) / len(words) / 100
            left = min(word[2] for word in words)
            top = min(word[3] for word in words)
            right = max(word[2] + word[4] for word in words)
            bottom = max(word[3] + word[5] for word in words)
            polygon = (
                (left, top),
                (right, top),
                (right, bottom),
                (left, bottom),
            )
            lines.append([polygon, (line_text, line_confidence)])
        return [lines]


class PDFConverter(Protocol):
    def convert(self, source: Any) -> Any: ...


class PdfExtractor:
    def __init__(
        self,
        converter: PDFConverter | None = None,
        ocr_reader: OCRReader | None = None,
        *,
        image_scale: float = _OCR_IMAGE_SCALE,
        ocr_engine: str | None = None,
        fusion_threshold: float | None = None,
    ) -> None:
        if image_scale <= 0:
            raise ValueError("image_scale must be greater than zero")
        self._converter = converter
        self._image_scale = image_scale
        engine = (ocr_engine or os.getenv("OCR_ENGINE", "paddleocr")).strip().lower()
        if engine not in {"paddleocr", "tesseract", "fused"}:
            raise ValueError(
                f"Unsupported OCR_ENGINE {engine!r}; choose 'paddleocr', "
                "'tesseract', or 'fused'"
            )
        self.ocr_engine = engine
        if fusion_threshold is None:
            fusion_threshold = (
                _read_fusion_threshold_env() if engine == "fused" else 90.0
            )
        if not 0 <= fusion_threshold <= 100:
            raise ValueError("fusion_threshold must be between 0 and 100")
        self.fusion_threshold = float(fusion_threshold)
        self._ocr_readers: dict[str, OCRReader] = {}
        if ocr_reader is not None:
            reader_engine = "tesseract" if engine == "fused" else engine
            self._ocr_readers[reader_engine] = ocr_reader

    def extract(self, source: Path | bytes, filename: str | None = None) -> PDFExtraction:
        extraction_started = perf_counter()
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

        native_pages, native_regions = self._extract_native_pages(
            document,
            page_items,
        )
        embedded_pdfs, image_pages = self._extract_pdf_resources(source, source_name)
        image_pages.update(self._get_docling_image_pages(document, page_items))
        pages_to_ocr = {
            page_number
            for page_number, page_text in native_pages.items()
            if assess_native_text(page_text)["ocr_recommended"]
        }
        pages_to_ocr.update(image_pages)
        pages_for_ocr = {
            page_number: page_items[page_number]
            for page_number in sorted(pages_to_ocr)
        }
        if self.ocr_engine == "fused":
            return self._extract_fused(
                pages_for_ocr=pages_for_ocr,
                native_pages=native_pages,
                native_regions=native_regions,
                embedded_pdfs=embedded_pdfs,
                page_count=page_count,
                pages_to_ocr=pages_to_ocr,
                source_name=source_name,
                extraction_started=extraction_started,
            )

        ocr_started = perf_counter()
        ocr_pages, ocr_confidence = self._extract_ocr_pages(
            pages_for_ocr,
            source_name,
            engine=self.ocr_engine,
        )
        ocr_time_seconds = (
            perf_counter() - ocr_started if pages_for_ocr else None
        )

        return self._assemble_extraction(
            native_pages=native_pages,
            native_regions=native_regions,
            embedded_pdfs=embedded_pdfs,
            page_count=page_count,
            pages_to_ocr=pages_to_ocr,
            ocr_pages=ocr_pages,
            ocr_confidence=ocr_confidence,
            ocr_engine=self.ocr_engine,
            ocr_time_seconds=ocr_time_seconds,
            extraction_started=extraction_started,
        )

    def _extract_fused(
        self,
        *,
        pages_for_ocr: dict[int, Any],
        native_pages: dict[int, str],
        native_regions: dict[int, list[NativeTextRegion]],
        embedded_pdfs: tuple[EmbeddedPDF, ...],
        page_count: int,
        pages_to_ocr: set[int],
        source_name: str,
        extraction_started: float,
    ) -> PDFExtraction:
        from app.fusion import fuse_ocr_lines

        tesseract_started = perf_counter()
        tesseract_lines, tesseract_confidence = self._extract_ocr_pages(
            pages_for_ocr,
            source_name,
            engine="tesseract",
        )
        tesseract_time = (
            perf_counter() - tesseract_started if pages_for_ocr else None
        )
        tesseract_extraction = self._assemble_extraction(
            native_pages=native_pages,
            native_regions=native_regions,
            embedded_pdfs=embedded_pdfs,
            page_count=page_count,
            pages_to_ocr=pages_to_ocr,
            ocr_pages=tesseract_lines,
            ocr_confidence=tesseract_confidence,
            ocr_engine="tesseract",
            ocr_time_seconds=tesseract_time,
            extraction_started=extraction_started,
        )
        tesseract_quality = self._assess_quality(tesseract_extraction)
        tesseract_page_quality = tesseract_quality.get("pages", {})
        pages_for_paddle = {
            page_number: page
            for page_number, page in pages_for_ocr.items()
            if (
                (page_quality := tesseract_page_quality.get(str(page_number), {})).get(
                    "score"
                )
                is None
                or float(page_quality["score"]) <= self.fusion_threshold
            )
        }
        if not pages_for_paddle:
            return replace(
                tesseract_extraction,
                ocr_engine="fused",
                quality_assessment=tesseract_quality,
                fusion_assessment={
                    "mode": "tesseract_first_paddle_fallback",
                    "threshold": self.fusion_threshold,
                    "threshold_comparison": (
                        "Run PaddleOCR when a Tesseract page quality score <= threshold"
                    ),
                    "tesseract_quality_score": tesseract_quality.get("score"),
                    "tesseract_quality_assessment": tesseract_quality,
                    "paddle_fallback_triggered": False,
                    "paddle_fallback_pages": [],
                    "selected_engine": "tesseract",
                },
            )

        paddle_started = perf_counter()
        paddle_lines, paddle_confidence = self._extract_ocr_pages(
            pages_for_paddle,
            source_name,
            engine="paddleocr",
        )
        paddle_time = perf_counter() - paddle_started
        paddle_extraction = self._assemble_extraction(
            native_pages=native_pages,
            native_regions=native_regions,
            embedded_pdfs=embedded_pdfs,
            page_count=page_count,
            pages_to_ocr=set(pages_for_paddle),
            ocr_pages=paddle_lines,
            ocr_confidence=paddle_confidence,
            ocr_engine="paddleocr",
            ocr_time_seconds=paddle_time,
            extraction_started=extraction_started,
        )
        paddle_quality = self._assess_quality(paddle_extraction)
        fused = fuse_ocr_lines(
            tesseract_lines,
            paddle_lines,
            native_regions,
            tesseract_quality=tesseract_quality,
            paddle_quality=paddle_quality,
            threshold=self.fusion_threshold,
        )
        fusion_assessment = dict(fused.assessment)
        fusion_assessment["paddle_fallback_triggered"] = True
        fusion_assessment["paddle_fallback_pages"] = sorted(pages_for_paddle)
        fusion_assessment["tesseract_quality_score"] = tesseract_quality.get("score")
        fusion_assessment["paddle_quality_score"] = paddle_quality.get("score")
        extraction = self._assemble_extraction(
            native_pages=native_pages,
            native_regions=native_regions,
            embedded_pdfs=embedded_pdfs,
            page_count=page_count,
            pages_to_ocr=pages_to_ocr,
            ocr_pages=fused.lines_by_page,
            ocr_confidence=fused.confidence_by_page,
            ocr_engine="fused",
            ocr_time_seconds=(tesseract_time or 0.0) + paddle_time,
            extraction_started=extraction_started,
            fusion_assessment=fusion_assessment,
        )
        combined_quality = self._assess_quality(extraction)
        combined_quality["method"] = "mixed_engine_heuristic_v1"
        combined_quality["note"] = (
            "This score is based on fused lines and mixed engine confidences; "
            "compare it cautiously with single-engine scores."
        )
        return replace(extraction, quality_assessment=combined_quality)

    @staticmethod
    def _assess_quality(extraction: PDFExtraction) -> dict[str, Any]:
        from app.postprocessor import review_flags_by_page
        from app.quality import assess_extraction_quality

        return assess_extraction_quality(
            extraction,
            review_flags_by_page(extraction.ocr_lines),
        )

    def _assemble_extraction(
        self,
        *,
        native_pages: dict[int, str],
        native_regions: dict[int, list[NativeTextRegion]],
        embedded_pdfs: tuple[EmbeddedPDF, ...],
        page_count: int,
        pages_to_ocr: set[int],
        ocr_pages: dict[int, tuple[OCRLine, ...]],
        ocr_confidence: dict[int, float],
        ocr_engine: str,
        ocr_time_seconds: float | None,
        extraction_started: float,
        fusion_assessment: dict[str, Any] | None = None,
    ) -> PDFExtraction:
        native_page_assessments = {
            str(page_number): assess_native_text(native_text)
            for page_number, native_text in native_pages.items()
        }
        ocr_recommended_pages = [
            int(page_number)
            for page_number, assessment in native_page_assessments.items()
            if assessment["ocr_recommended"]
        ]
        native_text_assessment = {
            "method": "native_text_heuristic_v1",
            "ocr_recommended": bool(ocr_recommended_pages),
            "ocr_recommended_pages": ocr_recommended_pages,
            "pages": native_page_assessments,
        }
        page_texts: dict[int, str] = {}
        for page_number, native_text in native_pages.items():
            recognized_lines = ocr_pages.get(page_number, ())
            page_assessment = native_page_assessments[str(page_number)]
            if not recognized_lines:
                page_texts[page_number] = native_text
            elif page_assessment["ocr_recommended"]:
                page_texts[page_number] = self._format_ocr_lines(recognized_lines)
            else:
                supplemental_lines = self._uncovered_ocr_lines(
                    recognized_lines,
                    native_text,
                    native_regions.get(page_number, ()),
                )
                if supplemental_lines:
                    page_texts[page_number] = self._join_text(
                        native_text,
                        self._format_ocr_supplement(supplemental_lines),
                    )
                else:
                    page_texts[page_number] = native_text

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
            ocr_lines=tuple(sorted(ocr_pages.items())),
            native_regions=tuple(
                (page_number, tuple(regions))
                for page_number, regions in sorted(native_regions.items())
            ),
            native_text_assessment=native_text_assessment,
            ocr_engine=ocr_engine,
            ocr_time_seconds=ocr_time_seconds,
            extraction_time_seconds=perf_counter() - extraction_started,
            fusion_assessment=fusion_assessment,
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
    def _extract_native_pages(
        document: Any,
        pages: dict[int, Any],
    ) -> tuple[dict[int, str], dict[int, list[NativeTextRegion]]]:
        page_text: dict[int, list[str]] = {page_number: [] for page_number in pages}
        page_regions: dict[int, list[NativeTextRegion]] = {
            page_number: [] for page_number in pages
        }
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
                    item_text = text.strip()
                    page_text[page_number].append(item_text)
                    page_regions[page_number].append(
                        NativeTextRegion(
                            text=item_text,
                            bbox=PdfExtractor._normalized_native_bbox(
                                getattr(source, "bbox", None),
                                pages[page_number],
                            ),
                        )
                    )

        return (
            {
                page_number: "\n".join(items)
                for page_number, items in page_text.items()
            },
            page_regions,
        )

    @staticmethod
    def _normalized_native_bbox(
        bbox: Any,
        page: Any,
    ) -> tuple[float, float, float, float] | None:
        if bbox is None:
            return None
        page_size = getattr(page, "size", None)
        try:
            page_width = float(page_size.width)
            page_height = float(page_size.height)
            left_value = float(bbox.l)
            right_value = float(bbox.r)
            top_value = float(bbox.t)
            bottom_value = float(bbox.b)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            return None
        if (
            page_width <= 0
            or page_height <= 0
            or not all(
                isfinite(value)
                for value in (
                    page_width,
                    page_height,
                    left_value,
                    right_value,
                    top_value,
                    bottom_value,
                )
            )
        ):
            return None

        left = left_value / page_width
        right = right_value / page_width

        origin = getattr(bbox, "coord_origin", "TOPLEFT")
        origin = str(getattr(origin, "value", origin)).upper()
        if origin == "BOTTOMLEFT":
            top = (page_height - top_value) / page_height
            bottom = (page_height - bottom_value) / page_height
        else:
            top = top_value / page_height
            bottom = bottom_value / page_height

        left, right = sorted((max(0.0, min(1.0, left)), max(0.0, min(1.0, right))))
        top, bottom = sorted((max(0.0, min(1.0, top)), max(0.0, min(1.0, bottom))))
        if right <= left or bottom <= top:
            return None
        return left, top, right, bottom

    @staticmethod
    def _get_docling_image_pages(
        document: Any,
        pages: dict[int, Any],
    ) -> set[int]:
        image_pages: set[int] = set()
        for item, _ in document.iterate_items():
            label = getattr(item, "label", None)
            if getattr(label, "value", label) != "picture":
                continue

            provenance = getattr(item, "prov", ())
            if not provenance:
                return set(pages)
            for source in provenance:
                page_number = int(source.page_no)
                if page_number in pages:
                    image_pages.add(page_number)
        return image_pages

    @staticmethod
    def _native_text_is_suspicious(text: str) -> bool:
        return bool(assess_native_text(text)["ocr_recommended"])

    @staticmethod
    def _uncovered_ocr_lines(
        ocr_lines: tuple[OCRLine, ...],
        native_text: str,
        native_regions: list[NativeTextRegion] | tuple[NativeTextRegion, ...],
    ) -> tuple[OCRLine, ...]:
        reference_tokens = [
            token.casefold() for token in _WORD_PATTERN.findall(native_text)
        ]
        reference_compact = "".join(
            character.casefold()
            for character in native_text
            if character.isalnum()
        )
        uncovered: list[OCRLine] = []
        for line in ocr_lines:
            line_tokens = [
                token.casefold() for token in _WORD_PATTERN.findall(line.text)
            ]
            compact_line = "".join(
                character.casefold()
                for character in line.text
                if character.isalnum()
            )
            if not compact_line:
                continue
            line_bbox = _polygon_bbox(line.polygon)
            if line_bbox is not None:
                if any(
                    region.bbox is not None
                    and _intersection_over_self(line_bbox, region.bbox) >= 0.5
                    and _similar_text_in_region(line_tokens, region.text)
                    for region in native_regions
                ):
                    continue
            elif (
                len(compact_line) >= 8
                and compact_line in reference_compact
            ) or _contains_token_sequence(reference_tokens, line_tokens):
                continue
            uncovered.append(line)
        return tuple(uncovered)

    @staticmethod
    def _format_ocr_lines(lines: tuple[OCRLine, ...]) -> str:
        return "\n".join(
            ["[OCR line positions are normalized to page size]"]
            + [PdfExtractor._format_ocr_line(line) for line in lines]
        )

    @staticmethod
    def _format_ocr_supplement(lines: tuple[OCRLine, ...]) -> str:
        formatted = [
            "[OCR text not represented in the PDF text layer; positions are "
            "normalized to page size]"
        ]
        formatted.extend(PdfExtractor._format_ocr_line(line) for line in lines)
        return "\n".join(formatted)

    @staticmethod
    def _format_ocr_line(line: OCRLine) -> str:
        bbox = _polygon_bbox(line.polygon)
        if bbox is None:
            location = "[position unavailable]"
        else:
            left, top, right, bottom = bbox
            location = f"[x={left:.3f}..{right:.3f}, y={top:.3f}..{bottom:.3f}]"
        return f"{location} {line.text}"

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
        *,
        engine: str,
    ) -> tuple[dict[int, tuple[OCRLine, ...]], dict[int, float]]:
        if not pages:
            return {}, {}
        reader = self._get_ocr_reader(engine)
        result: dict[int, tuple[OCRLine, ...]] = {}
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
            image = image.convert("RGB")
            lines, page_confidence = self._recognized_text(
                reader.ocr(np.asarray(image), cls=True),
                page_number,
                image.width,
                image.height,
            )
            result[page_number] = lines
            if page_confidence is not None:
                confidence[page_number] = page_confidence
        return result, confidence

    def _get_ocr_reader(self, engine: str) -> OCRReader:
        if engine == "fused":
            raise ValueError("The fused mode requires a specific OCR engine per pass")
        reader = self._ocr_readers.get(engine)
        if reader is not None:
            return reader
        if engine == "tesseract":
            try:
                import pytesseract
            except ImportError as error:
                raise ExtractionError(
                    "pytesseract is required for the Tesseract OCR engine; "
                    "install project requirements"
                ) from error
            reader = TesseractOCRReader(pytesseract)
        elif engine == "paddleocr":
            try:
                # PaddlePaddle 2.6 loads a bundled zlib symbol that can break
                # later imports of SciPy's native modules on ARM64.
                import scipy  # noqa: F401
                from paddleocr import PaddleOCR
            except ImportError as error:
                raise ExtractionError(
                    "PaddleOCR is required for scanned PDFs; install project requirements"
                ) from error
            reader = PaddleOCR(use_angle_cls=True, lang="en")
        else:
            raise ValueError(f"Unsupported OCR engine {engine!r}")
        self._ocr_readers[engine] = reader
        return reader

    @staticmethod
    def _recognized_text(
        raw_result: Any,
        page_number: int,
        image_width: int,
        image_height: int,
    ) -> tuple[tuple[OCRLine, ...], float | None]:
        if not isinstance(raw_result, list):
            raise ExtractionError(
                f"PaddleOCR returned an invalid result for page {page_number}"
            )
        if not raw_result or raw_result[0] is None:
            return (), None

        lines: list[OCRLine] = []
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
                line_confidence: float | None = None
                try:
                    line_confidence = float(line[1][1])
                    confidence_scores.append(line_confidence)
                except (IndexError, TypeError, ValueError):
                    pass
                lines.append(
                    OCRLine(
                        text=recognized_text.strip(),
                        confidence=line_confidence,
                        polygon=_normalized_ocr_polygon(
                            line[0] if isinstance(line, (list, tuple)) else None,
                            image_width,
                            image_height,
                        ),
                    )
                )
        mean_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else None
        )
        return tuple(lines), mean_confidence

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


def _normalized_ocr_polygon(
    raw_polygon: Any,
    image_width: int,
    image_height: int,
) -> tuple[tuple[float, float], ...] | None:
    if raw_polygon is None or image_width <= 0 or image_height <= 0:
        return None
    try:
        points = tuple(
            (
                max(0.0, min(1.0, float(point[0]) / image_width)),
                max(0.0, min(1.0, float(point[1]) / image_height)),
            )
            for point in raw_polygon
        )
    except (IndexError, TypeError, ValueError):
        return None
    return (
        points
        if len(points) >= 3
        and all(isfinite(value) for point in points for value in point)
        else None
    )


def _polygon_bbox(
    polygon: tuple[tuple[float, float], ...] | None,
) -> tuple[float, float, float, float] | None:
    if not polygon:
        return None
    x_values = [point[0] for point in polygon]
    y_values = [point[1] for point in polygon]
    left, right = min(x_values), max(x_values)
    top, bottom = min(y_values), max(y_values)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _intersection_over_self(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    return intersection / first_area if first_area else 0.0


def _contains_token_sequence(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    needle_length = len(needle)
    return any(
        haystack[index : index + needle_length] == needle
        for index in range(len(haystack) - needle_length + 1)
    )


def _similar_text_in_region(needle: list[str], region_text: str) -> bool:
    """Check whether a nearby native text region likely contains this OCR line."""
    if not needle:
        return False
    haystack = [token.casefold() for token in _WORD_PATTERN.findall(region_text)]
    if _contains_token_sequence(haystack, needle):
        return True
    if len("".join(needle)) < 6:
        return False
    needle_text = " ".join(needle)
    min_window = max(1, len(needle) - 1)
    max_window = min(len(haystack), len(needle) + 1)
    for window_size in range(min_window, max_window + 1):
        for start in range(len(haystack) - window_size + 1):
            candidate = haystack[start : start + window_size]
            similarity = SequenceMatcher(
                None,
                needle_text,
                " ".join(candidate),
            ).ratio()
            if similarity >= 0.82:
                return True
    return False


def _is_pdf_attachment(name: str, content: bytes) -> bool:
    return name.lower().endswith(".pdf") or b"%PDF-" in content[:1024]


def _read_fusion_threshold_env() -> float:
    raw_value = os.getenv("OCR_FUSION_THRESHOLD", "90")
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError("OCR_FUSION_THRESHOLD must be a number from 0 to 100") from error
    if not 0 <= value <= 100:
        raise ValueError("OCR_FUSION_THRESHOLD must be between 0 and 100")
    return value


def _safe_pdf_filename(name: str, attachment_index: int) -> str:
    filename = PurePath(name.replace("\\", "/")).name
    if not filename:
        filename = f"embedded-{attachment_index}.pdf"
    elif not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"
    return filename
