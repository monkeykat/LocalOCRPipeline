from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from app.extractor import OCRLine

_POSITION_PREFIX = re.compile(
    r"^\[(?:x=(\d+(?:\.\d+)?)\.\.(\d+(?:\.\d+)?), "
    r"y=(\d+(?:\.\d+)?)\.\.(\d+(?:\.\d+)?)|position unavailable)\]\s*"
)
_OCR_SECTION_MARKERS = {
    "[OCR line positions are normalized to page size]",
    "[OCR text not represented in the PDF text layer; positions are normalized to page size]",
}
_SUSPICIOUS_TOKEN = re.compile(
    r"[A-Za-z][~`#$%&+<>;|][A-Za-z]|([A-Za-z])\1\1|"
    r"[A-Z][a-z]+[A-Z][a-z]"
)
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’._+/-]*")
_REVIEW_CONFIDENCE_THRESHOLD = 0.80


def clean_transcript(text: str) -> str:
    """Create a readable view while retaining page text and all recognized words."""
    text = unicodedata.normalize("NFC", text.replace("\u00a0", " "))
    output: list[str] = []
    in_ocr_section = False
    seen_ocr_lines: list[tuple[str, tuple[float, float, float, float] | None]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line in _OCR_SECTION_MARKERS:
            in_ocr_section = True
            seen_ocr_lines = []
            continue

        if line.startswith("=== PAGE ") and line.endswith(" ==="):
            in_ocr_section = False
            seen_ocr_lines = []

        position_match = _POSITION_PREFIX.match(line)
        bbox = _position_bbox(position_match)
        if position_match:
            line = line[position_match.end():]

        line = re.sub(r"[\t ]+", " ", line).strip()
        if in_ocr_section and line and _is_overlapping_duplicate(line, bbox, seen_ocr_lines):
            continue
        if in_ocr_section and line:
            seen_ocr_lines.append((_normalized_line(line), bbox))

        if not line:
            if output and output[-1] != "":
                output.append("")
            continue
        output.append(line)

    while output and not output[-1]:
        output.pop()
    return "\n".join(output) + ("\n" if output else "")


def review_flags_by_page(
    ocr_lines: Iterable[tuple[int, tuple[OCRLine, ...]]],
) -> dict[str, list[dict[str, Any]]]:
    """Flag low-confidence or visibly malformed OCR lines without rewriting them."""
    flags: dict[str, list[dict[str, Any]]] = {}
    for page_number, lines in ocr_lines:
        page_flags: list[dict[str, Any]] = []
        for line_index, line in enumerate(lines):
            reasons: list[str] = []
            suspicious_tokens = [
                token for token in _TOKEN.findall(line.text)
                if _SUSPICIOUS_TOKEN.search(token)
            ]
            if suspicious_tokens:
                reasons.append("suspicious_character_pattern")
            if (
                line.confidence is not None
                and line.confidence < _REVIEW_CONFIDENCE_THRESHOLD
            ):
                reasons.append("low_ocr_confidence")
            if not reasons:
                continue

            flag: dict[str, Any] = {
                "line_index": line_index,
                "text": line.text,
                "reasons": reasons,
            }
            if suspicious_tokens:
                flag["suspicious_tokens"] = suspicious_tokens
            page_flags.append(flag)
        if page_flags:
            flags[str(page_number)] = page_flags
    return flags


def _position_bbox(
    match: re.Match[str] | None,
) -> tuple[float, float, float, float] | None:
    if match is None or match.group(1) is None:
        return None
    try:
        return tuple(float(match.group(index)) for index in range(1, 5))  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _normalized_line(line: str) -> str:
    return " ".join(line.casefold().split())


def _is_overlapping_duplicate(
    line: str,
    bbox: tuple[float, float, float, float] | None,
    previous: list[tuple[str, tuple[float, float, float, float] | None]],
) -> bool:
    normalized = _normalized_line(line)
    if not normalized or bbox is None:
        return False
    return any(
        normalized == old_text
        and old_bbox is not None
        and _intersection_over_smaller(bbox, old_bbox) >= 0.8
        for old_text, old_bbox in previous
    )


def _intersection_over_smaller(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    smaller_area = min(first_area, second_area)
    return intersection / smaller_area if smaller_area else 0.0
