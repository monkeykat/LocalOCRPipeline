"""Compare local OCR text and timing against each other and Azure output."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?")


@dataclass(frozen=True)
class OCRDocument:
    tokens: list[str]
    ocr_time_seconds: float | None
    extraction_time_seconds: float | None


def _read_seconds(metadata: dict[str, object], key: str) -> float | None:
    try:
        value = float(metadata[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _read_output_map(directory: Path) -> dict[str, OCRDocument]:
    documents: dict[str, OCRDocument] = {}
    if not directory.is_dir():
        return documents
    for path in sorted(directory.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = re.sub(r"(?im)^DOCUMENT:.*$", " ", text)
        text = re.sub(r"(?m)^=== PAGE \d+ ===$", " ", text)
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
        documents[key] = OCRDocument(
            tokens=_TOKEN_PATTERN.findall(text.casefold()),
            ocr_time_seconds=_read_seconds(metadata, "ocr_time_seconds"),
            extraction_time_seconds=_read_seconds(
                metadata,
                "extraction_time_seconds",
            ),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare normalized word order and recorded processing time for "
            "PaddleOCR, Tesseract, and an optional Azure reference."
        )
    )
    parser.add_argument("--paddle", type=Path, default=Path("data/output"))
    parser.add_argument(
        "--tesseract", type=Path, default=Path("data/output_tesseract")
    )
    parser.add_argument("--azure", type=Path, default=Path("data/doc_intel_output"))
    args = parser.parse_args()

    paddle = _read_output_map(args.paddle)
    tesseract = _read_output_map(args.tesseract)
    azure = _read_output_map(args.azure)
    keys = sorted(paddle.keys() | tesseract.keys())
    if not keys:
        parser.error("No .txt outputs found in the PaddleOCR or Tesseract folders")

    print(
        "Text similarity (%)\n"
        "Document | Paddle words | Tesseract words | Paddle↔Tesseract | "
        "Paddle↔Azure | Tesseract↔Azure"
    )
    print("---------|--------------|-----------------|-----------------|--------------|----------------")
    for key in keys:
        paddle_doc = paddle.get(key)
        tesseract_doc = tesseract.get(key)
        print(
            f"{key} | {len(paddle_doc.tokens) if paddle_doc else '—'} | "
            f"{len(tesseract_doc.tokens) if tesseract_doc else '—'} | "
            f"{_similarity(paddle_doc, tesseract_doc)} | "
            f"{_similarity(paddle_doc, azure.get(key))} | "
            f"{_similarity(tesseract_doc, azure.get(key))}"
        )

    print(
        "\nProcessing time (seconds)\n"
        "Document | Paddle OCR | Tesseract OCR | OCR speedup | "
        "Paddle extraction | Tesseract extraction"
    )
    print("---------|------------|---------------|-------------|------------------|---------------------")
    for key in keys:
        paddle_doc = paddle.get(key)
        tesseract_doc = tesseract.get(key)
        paddle_ocr = paddle_doc.ocr_time_seconds if paddle_doc else None
        tesseract_ocr = tesseract_doc.ocr_time_seconds if tesseract_doc else None
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

    print(
        "\nText percentages compare normalized token order after removing page labels, "
        "case, and punctuation. Azure output is a reference, not guaranteed ground truth. "
        "OCR time includes engine initialization and OCR calls; extraction time also "
        "includes PDF parsing/rendering but excludes Docker startup and output writing."
    )


if __name__ == "__main__":
    main()
