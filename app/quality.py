from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from app.extractor import NativeTextRegion, OCRLine, PDFExtraction

_QUALITY_REVIEW_THRESHOLD = 85.0
_QUALITY_WEIGHTS = {
    "mean_ocr_confidence": 0.45,
    "native_text_agreement": 0.35,
    "line_integrity": 0.20,
}
_WORD = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?")


def assess_extraction_quality(
    extraction: PDFExtraction,
    review_flags: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build a transparent, per-page quality triage score, not an accuracy claim."""
    ocr_lines = dict(extraction.ocr_lines)
    native_regions = dict(extraction.native_regions)
    confidence_by_page = dict(extraction.ocr_confidence)
    ocr_pages = extraction.ocr_pages
    page_results: dict[str, dict[str, Any]] = {}
    scored_pages: list[float] = []

    for page_number in ocr_pages:
        lines = ocr_lines.get(page_number, ())
        if not lines:
            page_result = {
                "score": 0.0,
                "band": "low",
                "line_count": 0,
                "mean_ocr_confidence": 0.0,
                "native_text_agreement": None,
                "compared_line_count": 0,
                "suspicious_line_count": 0,
                "review_recommended": True,
            }
            page_results[str(page_number)] = page_result
            scored_pages.append(0.0)
            continue

        confidence_scores = [
            line.confidence for line in lines if line.confidence is not None
        ]
        mean_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else confidence_by_page.get(page_number)
        )
        suspicious_count = sum(
            "suspicious_character_pattern" in flag.get("reasons", ())
            for flag in review_flags.get(str(page_number), ())
        )
        agreement, compared_line_count = _native_ocr_agreement(
            lines,
            native_regions.get(page_number, ()),
        )

        components: list[tuple[float, float]] = []
        if mean_confidence is not None:
            components.append(
                (_QUALITY_WEIGHTS["mean_ocr_confidence"], mean_confidence * 100)
            )
        if agreement is not None:
            components.append(
                (_QUALITY_WEIGHTS["native_text_agreement"], agreement * 100)
            )
        line_integrity = 1 - suspicious_count / len(lines)
        components.append(
            (_QUALITY_WEIGHTS["line_integrity"], line_integrity * 100)
        )
        weight_total = sum(weight for weight, _ in components)
        score = sum(weight * value for weight, value in components) / weight_total
        score = max(0.0, min(100.0, score))
        page_result = {
            "score": round(score, 1),
            "band": _quality_band(score),
            "line_count": len(lines),
            "mean_ocr_confidence": (
                round(mean_confidence, 4) if mean_confidence is not None else None
            ),
            "native_text_agreement": (
                round(agreement, 4) if agreement is not None else None
            ),
            "compared_line_count": compared_line_count,
            "suspicious_line_count": suspicious_count,
            "review_recommended": score < _QUALITY_REVIEW_THRESHOLD,
        }
        page_results[str(page_number)] = page_result
        scored_pages.append(score)

    if not scored_pages:
        return {
            "score": None,
            "band": "unrated",
            "review_recommended": False,
            "review_threshold": _QUALITY_REVIEW_THRESHOLD,
            "method": "heuristic_v1",
            "reason": "No OCR pages were available to score",
            "pages": {},
        }

    mean_page_score = sum(scored_pages) / len(scored_pages)
    weakest_page_score = min(scored_pages)
    document_score = 0.75 * mean_page_score + 0.25 * weakest_page_score
    document_score = max(0.0, min(100.0, document_score))
    return {
        "score": round(document_score, 1),
        "band": _quality_band(document_score),
        "review_recommended": document_score < _QUALITY_REVIEW_THRESHOLD,
        "review_threshold": _QUALITY_REVIEW_THRESHOLD,
        "method": "heuristic_v1",
        "scored_page_count": len(scored_pages),
        "mean_page_score": round(mean_page_score, 1),
        "weakest_page_score": round(weakest_page_score, 1),
        "pages": page_results,
    }


def _quality_band(score: float) -> str:
    if score >= 90:
        return "high"
    if score >= 75:
        return "review"
    return "low"


def _native_ocr_agreement(
    lines: tuple[OCRLine, ...],
    regions: tuple[NativeTextRegion, ...],
) -> tuple[float | None, int]:
    if not regions:
        return None, 0
    region_tokens = [
        (_region_bbox(region), _tokens(region.text))
        for region in regions
        if region.bbox is not None and _tokens(region.text)
    ]
    if not region_tokens:
        return None, 0

    line_scores: list[float] = []
    for line in lines:
        line_bbox = _line_bbox(line)
        line_tokens = _tokens(line.text)
        if line_bbox is None or not line_tokens:
            continue
        overlaps = [
            tokens
            for region_bbox, tokens in region_tokens
            if _overlap_of_first(line_bbox, region_bbox) >= 0.35
        ]
        if not overlaps:
            continue
        line_scores.append(
            max(_token_coverage(line_tokens, tokens) for tokens in overlaps)
        )
    if not line_scores:
        return None, 0
    return sum(line_scores) / len(line_scores), len(line_scores)


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


def _token_coverage(first: list[str], second: list[str]) -> float:
    matched = 0
    for token in first:
        if any(
            token == candidate
            or SequenceMatcher(None, token, candidate).ratio() >= 0.82
            for candidate in second
        ):
            matched += 1
    return matched / len(first) if first else 0.0


def _line_bbox(
    line: OCRLine,
) -> tuple[float, float, float, float] | None:
    if not line.polygon:
        return None
    return (
        min(point[0] for point in line.polygon),
        min(point[1] for point in line.polygon),
        max(point[0] for point in line.polygon),
        max(point[1] for point in line.polygon),
    )


def _region_bbox(
    region: NativeTextRegion,
) -> tuple[float, float, float, float] | None:
    return region.bbox


def _overlap_of_first(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    return intersection / area if area else 0.0
