"""Compare local OCR text, runtime, and quality ranking against Azure output."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?")
_POSITION_PREFIX = re.compile(
    r"^\[(?:x=\d+(?:\.\d+)?\.\.\d+(?:\.\d+)?, "
    r"y=\d+(?:\.\d+)?\.\.\d+(?:\.\d+)?|position unavailable)\]\s*"
)
_OCR_POSITION_HEADERS = {
    "[OCR line positions are normalized to page size]",
    "[OCR text not represented in the PDF text layer; positions are normalized to page size]",
}


@dataclass(frozen=True)
class OCRDocument:
    tokens: list[str]
    ocr_time_seconds: float | None
    extraction_time_seconds: float | None
    quality_score: float | None
    quality_band: str | None
    review_recommended: bool | None


def _read_seconds(metadata: dict[str, object], key: str) -> float | None:
    try:
        value = float(metadata[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _comparison_text(text: str) -> str:
    """Remove document layout annotations while retaining recognized text."""
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line in _OCR_POSITION_HEADERS:
            continue
        if re.match(r"(?i)^DOCUMENT:", line) or re.fullmatch(
            r"(?i)=== PAGE \d+ ===", line
        ):
            continue
        line = _POSITION_PREFIX.sub("", line).strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _read_output_map(directory: Path) -> dict[str, OCRDocument]:
    documents: dict[str, OCRDocument] = {}
    if not directory.is_dir():
        return documents
    for path in sorted(directory.rglob("*.txt")):
        if path.name.endswith(".cleaned.txt"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = _comparison_text(text)
        key = path.stem.casefold()
        if key in documents:
            key = path.relative_to(directory).with_suffix("").as_posix().casefold()
        try:
            metadata = json.loads(
                path.with_suffix(".json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        assessment = metadata.get("quality_assessment")
        if not isinstance(assessment, dict):
            assessment = {}
        try:
            quality_score = float(assessment["score"])
        except (KeyError, TypeError, ValueError):
            quality_score = None
        quality_band = assessment.get("band")
        if not isinstance(quality_band, str):
            quality_band = None
        review_recommended = assessment.get("review_recommended")
        if not isinstance(review_recommended, bool):
            review_recommended = None
        documents[key] = OCRDocument(
            tokens=_TOKEN_PATTERN.findall(text.casefold()),
            ocr_time_seconds=_read_seconds(metadata, "ocr_time_seconds"),
            extraction_time_seconds=_read_seconds(
                metadata,
                "extraction_time_seconds",
            ),
            quality_score=quality_score,
            quality_band=quality_band,
            review_recommended=review_recommended,
        )
    return documents


def _similarity(first: OCRDocument | None, second: OCRDocument | None) -> str:
    if first is None or second is None:
        return "—"
    if not first.tokens and not second.tokens:
        return "100.0%"
    score = SequenceMatcher(None, first.tokens, second.tokens).ratio() * 100
    return f"{score:.1f}%"


def _format_seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _format_speedup(first: float | None, second: float | None) -> str:
    if first is None or second is None or second <= 0:
        return "—"
    return f"{first / second:.2f}x"


def _print_quality_ranking(name: str, documents: dict[str, OCRDocument]) -> None:
    rated = sorted(
        (
            (key, document)
            for key, document in documents.items()
            if document.quality_score is not None
        ),
        key=lambda item: (item[1].quality_score, item[0]),
    )
    if not rated:
        return
    print(f"\nQuality ranking, lowest score first ({name})")
    print("Rank | Document | Score | Band | Review")
    print("-----|----------|-------|------|-------")
    for rank, (key, document) in enumerate(rated, start=1):
        review = "yes" if document.review_recommended else "no"
        print(
            f"{rank} | {key} | {document.quality_score:.1f} | "
            f"{document.quality_band or 'unknown'} | {review}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare normalized token order and recorded processing time for "
            "PaddleOCR and Tesseract, rank heuristic quality scores, and use "
            "an optional Azure reference."
        )
    )
    parser.add_argument(
        "--paddle", type=Path, default=Path("data/output_PaddleOCR")
    )
    parser.add_argument(
        "--tesseract", type=Path, default=Path("data/output_tesseract")
    )
    parser.add_argument("--fused", type=Path)
    parser.add_argument("--azure", type=Path, default=Path("data/doc_intel_output"))
    args = parser.parse_args()

    paddle = _read_output_map(args.paddle)
    tesseract = _read_output_map(args.tesseract)
    fused = _read_output_map(args.fused) if args.fused else {}
    azure = _read_output_map(args.azure)
    keys = sorted(paddle.keys() | tesseract.keys() | fused.keys())
    if not keys:
        parser.error("No .txt outputs found in the selected OCR output folders")

    if args.fused:
        print(
            "Text similarity (%)\n"
            "Document | Paddle tokens | Tesseract tokens | Fused tokens | "
            "Paddle↔Tesseract | Paddle↔Azure | Tesseract↔Azure | Fused↔Azure | "
            "Paddle↔Fused | Tesseract↔Fused"
        )
        print(
            "---------|--------------|-----------------|-------------|-----------------"
            "|--------------|-----------------|-------------|--------------|-----------------"
        )
    else:
        print(
            "Text similarity (%)\n"
            "Document | Paddle tokens | Tesseract tokens | Paddle↔Tesseract | "
            "Paddle↔Azure | Tesseract↔Azure"
        )
        print(
            "---------|--------------|-----------------|-----------------|"
            "--------------|----------------"
        )
    for key in keys:
        paddle_doc = paddle.get(key)
        tesseract_doc = tesseract.get(key)
        if args.fused:
            fused_doc = fused.get(key)
            print(
                f"{key} | {len(paddle_doc.tokens) if paddle_doc else '—'} | "
                f"{len(tesseract_doc.tokens) if tesseract_doc else '—'} | "
                f"{len(fused_doc.tokens) if fused_doc else '—'} | "
                f"{_similarity(paddle_doc, tesseract_doc)} | "
                f"{_similarity(paddle_doc, azure.get(key))} | "
                f"{_similarity(tesseract_doc, azure.get(key))} | "
                f"{_similarity(fused_doc, azure.get(key))} | "
                f"{_similarity(paddle_doc, fused_doc)} | "
                f"{_similarity(tesseract_doc, fused_doc)}"
            )
        else:
            print(
                f"{key} | {len(paddle_doc.tokens) if paddle_doc else '—'} | "
                f"{len(tesseract_doc.tokens) if tesseract_doc else '—'} | "
                f"{_similarity(paddle_doc, tesseract_doc)} | "
                f"{_similarity(paddle_doc, azure.get(key))} | "
                f"{_similarity(tesseract_doc, azure.get(key))}"
            )

    if args.fused:
        print(
            "\nProcessing time (seconds)\n"
            "Document | Paddle OCR | Tesseract OCR | Fused OCR | Paddle/Tesseract | "
            "Paddle extraction | Tesseract extraction | Fused extraction"
        )
        print(
            "---------|------------|---------------|-----------|-----------------"
            "|------------------|---------------------|-----------------"
        )
    else:
        print(
            "\nProcessing time (seconds)\n"
            "Document | Paddle OCR | Tesseract OCR | OCR speedup | "
            "Paddle extraction | Tesseract extraction"
        )
        print(
            "---------|------------|---------------|-------------|"
            "------------------|---------------------"
        )
    for key in keys:
        paddle_doc = paddle.get(key)
        tesseract_doc = tesseract.get(key)
        paddle_ocr = paddle_doc.ocr_time_seconds if paddle_doc else None
        tesseract_ocr = tesseract_doc.ocr_time_seconds if tesseract_doc else None
        if args.fused:
            fused_doc = fused.get(key)
            print(
                f"{key} | {_format_seconds(paddle_ocr)} | "
                f"{_format_seconds(tesseract_ocr)} | "
                f"{_format_seconds(fused_doc.ocr_time_seconds if fused_doc else None)} | "
                f"{_format_speedup(paddle_ocr, tesseract_ocr)} | "
                f"{_format_seconds(paddle_doc.extraction_time_seconds if paddle_doc else None)} | "
                f"{_format_seconds(tesseract_doc.extraction_time_seconds if tesseract_doc else None)} | "
                f"{_format_seconds(fused_doc.extraction_time_seconds if fused_doc else None)}"
            )
        else:
            print(
                f"{key} | {_format_seconds(paddle_ocr)} | "
                f"{_format_seconds(tesseract_ocr)} | "
                f"{_format_speedup(paddle_ocr, tesseract_ocr)} | "
                f"{_format_seconds(paddle_doc.extraction_time_seconds if paddle_doc else None)} | "
                f"{_format_seconds(tesseract_doc.extraction_time_seconds if tesseract_doc else None)}"
            )

    common_ocr_times = [
        (paddle[key].ocr_time_seconds, tesseract[key].ocr_time_seconds)
        for key in keys
        if key in paddle
        and key in tesseract
        and paddle[key].ocr_time_seconds is not None
        and tesseract[key].ocr_time_seconds is not None
    ]
    common_extraction_times = [
        (paddle[key].extraction_time_seconds, tesseract[key].extraction_time_seconds)
        for key in keys
        if key in paddle
        and key in tesseract
        and paddle[key].extraction_time_seconds is not None
        and tesseract[key].extraction_time_seconds is not None
    ]
    if common_ocr_times:
        paddle_ocr_total = sum(pair[0] for pair in common_ocr_times)
        tesseract_ocr_total = sum(pair[1] for pair in common_ocr_times)
        print(
            f"\nOCR totals for {len(common_ocr_times)} documents timed by both engines: "
            f"Paddle {paddle_ocr_total:.2f}s; "
            f"Tesseract {tesseract_ocr_total:.2f}s; "
            f"speedup {_format_speedup(paddle_ocr_total, tesseract_ocr_total)}"
        )
    if common_extraction_times:
        paddle_extraction_total = sum(pair[0] for pair in common_extraction_times)
        tesseract_extraction_total = sum(pair[1] for pair in common_extraction_times)
        print(
            f"Full extraction totals for {len(common_extraction_times)} documents: "
            f"Paddle {paddle_extraction_total:.2f}s; "
            f"Tesseract {tesseract_extraction_total:.2f}s; "
            f"speedup {_format_speedup(paddle_extraction_total, tesseract_extraction_total)}"
        )

    if args.fused:
        fused_ocr_times = [
            fused[key].ocr_time_seconds
            for key in keys
            if key in fused and fused[key].ocr_time_seconds is not None
        ]
        fused_extraction_times = [
            fused[key].extraction_time_seconds
            for key in keys
            if key in fused and fused[key].extraction_time_seconds is not None
        ]
        if fused_ocr_times:
            print(
                f"Fused OCR total for {len(fused_ocr_times)} documents: "
                f"{sum(fused_ocr_times):.2f}s"
            )
        if fused_extraction_times:
            print(
                f"Fused extraction total for {len(fused_extraction_times)} documents: "
                f"{sum(fused_extraction_times):.2f}s"
            )

    _print_quality_ranking("PaddleOCR", paddle)
    _print_quality_ranking("Tesseract", tesseract)
    if args.fused:
        _print_quality_ranking("Fused", fused)

    print(
        "\nText percentages compare normalized token order after removing document/page "
        "headings, OCR position annotations, case, and punctuation. Azure output is a "
        "reference, not guaranteed ground truth. "
        "OCR time includes engine initialization and OCR calls; extraction time also "
        "includes PDF parsing/rendering but excludes Docker startup and output writing. "
        "Quality scores are heuristic triage ranks and should be compared within an "
        "engine, not across engines."
    )


if __name__ == "__main__":
    main()
