from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from app.extractor import NativeTextRegion, OCRLine

_TOKEN = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?", re.IGNORECASE)
_SUSPICIOUS = re.compile(
    r"[A-Za-z][~`#$%&+<>;|][A-Za-z]|([A-Za-z])\1\1|"
    r"[A-Z][a-z]+[A-Z][a-z]"
)
_PAGE_ENGINE_MIN_SCORE = 85.0
_PAGE_ENGINE_SCORE_MARGIN = 8.0


@dataclass(frozen=True)
class FusedOCR:
    lines_by_page: dict[int, tuple[OCRLine, ...]]
    confidence_by_page: dict[int, float]
    assessment: dict[str, Any]


def fuse_ocr_lines(
    tesseract_lines: dict[int, tuple[OCRLine, ...]],
    paddle_lines: dict[int, tuple[OCRLine, ...]],
    native_regions: dict[int, list[NativeTextRegion]],
    *,
    tesseract_quality: dict[str, Any],
    paddle_quality: dict[str, Any],
    threshold: float,
) -> FusedOCR:
    """Merge page-local OCR lines and record ambiguous alternatives."""
    result: dict[int, tuple[OCRLine, ...]] = {}
    confidences: dict[int, float] = {}
    page_details: dict[str, dict[str, Any]] = {}
    conflicts: dict[str, list[dict[str, Any]]] = {}
    sources_by_page: dict[str, list[str]] = {}

    for page_number in sorted(tesseract_lines.keys() | paddle_lines.keys()):
        tess = list(tesseract_lines.get(page_number, ()))
        paddle = list(paddle_lines.get(page_number, ()))
        regions = native_regions.get(page_number, [])
        pairs = _match_lines(tess, paddle)
        preferred_engine, tess_page_score, paddle_page_score = _preferred_page_engine(
            page_number,
            tesseract_quality,
            paddle_quality,
        )
        if preferred_engine is not None:
            preferred_lines = tess if preferred_engine == "tesseract" else paddle
            selected_engine = (
                "tesseract" if preferred_engine == "tesseract" else "paddleocr"
            )
            result[page_number] = tuple(preferred_lines)
            sources_by_page[str(page_number)] = [selected_engine] * len(preferred_lines)
            page_confidence = [
                line.confidence
                for line in preferred_lines
                if line.confidence is not None
            ]
            if page_confidence:
                confidences[page_number] = sum(page_confidence) / len(page_confidence)

            page_conflicts: list[dict[str, Any]] = []
            agreed_lines = 0
            for tess_index, paddle_index in pairs:
                tess_line = tess[tess_index]
                paddle_line = paddle[paddle_index]
                if _same_text(tess_line.text, paddle_line.text):
                    agreed_lines += 1
                    continue
                preferred_line = tess_line if preferred_engine == "tesseract" else paddle_line
                alternate_line = paddle_line if preferred_engine == "tesseract" else tess_line
                page_conflicts.append(
                    {
                        "selected_engine": selected_engine,
                        "selected_text": preferred_line.text,
                        "alternate_engine": (
                            "paddleocr" if preferred_engine == "tesseract" else "tesseract"
                        ),
                        "alternate_text": alternate_line.text,
                        "reason": "higher_page_quality_score",
                        "tesseract_line_index": tess_index,
                        "paddle_line_index": paddle_index,
                        **_bbox_metadata(preferred_line),
                    }
                )
            page_details[str(page_number)] = {
                "tesseract_line_count": len(tess),
                "paddle_line_count": len(paddle),
                "matched_line_count": len(pairs),
                "conflict_count": len(page_conflicts),
                "selected_from_tesseract": (
                    len(preferred_lines) if preferred_engine == "tesseract" else 0
                ),
                "selected_from_paddle": (
                    len(preferred_lines) if preferred_engine == "paddleocr" else 0
                ),
                "agreed_between_engines": agreed_lines,
                "page_selection": f"{selected_engine}_higher_quality",
                "tesseract_page_quality_score": tess_page_score,
                "paddle_page_quality_score": paddle_page_score,
            }
            if page_conflicts:
                conflicts[str(page_number)] = page_conflicts
            continue

        matched_tess = {left for left, _ in pairs}
        matched_paddle = {right for _, right in pairs}
        selected: list[tuple[OCRLine, str, int]] = []
        page_conflicts: list[dict[str, Any]] = []
        source_counts = {"tesseract": 0, "paddleocr": 0, "both": 0}

        for tess_index, paddle_index in pairs:
            tess_line = tess[tess_index]
            paddle_line = paddle[paddle_index]
            same_text = _same_text(tess_line.text, paddle_line.text)
            winner, reason = _choose_line(tess_line, paddle_line, regions)
            selected_engine = (
                "both" if same_text else ("tesseract" if winner is tess_line else "paddleocr")
            )
            source_counts[selected_engine] += 1
            selected.append(
                (winner, selected_engine, min(tess_index, paddle_index))
            )
            if not same_text:
                page_conflicts.append(
                    {
                        "selected_engine": selected_engine,
                        "selected_text": winner.text,
                        "alternate_engine": (
                            "paddleocr" if winner is tess_line else "tesseract"
                        ),
                        "alternate_text": (
                            paddle_line.text if winner is tess_line else tess_line.text
                        ),
                        "reason": reason,
                        "tesseract_line_index": tess_index,
                        "paddle_line_index": paddle_index,
                        **_bbox_metadata(winner),
                    }
                )

        for index, line in enumerate(tess):
            if index not in matched_tess:
                source_counts["tesseract"] += 1
                selected.append((line, "tesseract", index))
        for index, line in enumerate(paddle):
            if index not in matched_paddle:
                source_counts["paddleocr"] += 1
                selected.append((line, "paddleocr", index))

        selected.sort(key=lambda item: _reading_order(item[0], item[2]))
        result[page_number] = tuple(line for line, _, _ in selected)
        sources_by_page[str(page_number)] = [source for _, source, _ in selected]
        page_details[str(page_number)] = {
            "tesseract_line_count": len(tess),
            "paddle_line_count": len(paddle),
            "matched_line_count": len(pairs),
            "tesseract_only_line_count": len(tess) - len(matched_tess),
            "paddle_only_line_count": len(paddle) - len(matched_paddle),
            "conflict_count": len(page_conflicts),
            "selected_from_tesseract": source_counts["tesseract"],
            "selected_from_paddle": source_counts["paddleocr"],
            "agreed_between_engines": source_counts["both"],
        }
        if page_conflicts:
            conflicts[str(page_number)] = page_conflicts

        page_confidence = [
            line.confidence
            for line, _, _ in selected
            if line.confidence is not None
        ]
        if page_confidence:
            confidences[page_number] = sum(page_confidence) / len(page_confidence)

    assessment: dict[str, Any] = {
        "mode": "tesseract_first_paddle_fallback",
        "fusion_algorithm_version": 2,
        "threshold": threshold,
        "threshold_comparison": "Tesseract page quality score <= threshold",
        "tesseract_quality_assessment": tesseract_quality,
        "paddle_quality_assessment": paddle_quality,
        "selection_strategy": (
            "When one engine's page quality score is at least "
            f"{_PAGE_ENGINE_MIN_SCORE:.0f} and leads by at least "
            f"{_PAGE_ENGINE_SCORE_MARGIN:.0f} points, use that engine's page output. "
            "Otherwise keep non-overlapping detections from both engines, collapse "
            "matched lines to one, prefer stronger native-text support, and record "
            "ambiguous alternatives for review."
        ),
        "page_quality_selection": {
            "minimum_score": _PAGE_ENGINE_MIN_SCORE,
            "minimum_lead": _PAGE_ENGINE_SCORE_MARGIN,
        },
        "page_counts": page_details,
        "line_sources_by_page": sources_by_page,
    }
    if conflicts:
        assessment["conflicts_by_page"] = conflicts
    return FusedOCR(result, confidences, assessment)


def _preferred_page_engine(
    page_number: int,
    tesseract_quality: dict[str, Any],
    paddle_quality: dict[str, Any],
) -> tuple[str | None, float | None, float | None]:
    tess_score = _page_quality_score(tesseract_quality, page_number)
    paddle_score = _page_quality_score(paddle_quality, page_number)
    if (
        paddle_score is not None
        and paddle_score >= _PAGE_ENGINE_MIN_SCORE
        and (tess_score is None or paddle_score - tess_score >= _PAGE_ENGINE_SCORE_MARGIN)
    ):
        return "paddleocr", tess_score, paddle_score
    if (
        tess_score is not None
        and tess_score >= _PAGE_ENGINE_MIN_SCORE
        and (paddle_score is None or tess_score - paddle_score >= _PAGE_ENGINE_SCORE_MARGIN)
    ):
        return "tesseract", tess_score, paddle_score
    return None, tess_score, paddle_score


def _page_quality_score(assessment: dict[str, Any], page_number: int) -> float | None:
    page = (assessment.get("pages") or {}).get(str(page_number))
    if not isinstance(page, dict):
        return None
    score = page.get("score")
    return float(score) if isinstance(score, (int, float)) else None


def _match_lines(
    tesseract: list[OCRLine],
    paddle: list[OCRLine],
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for tess_index, tess_line in enumerate(tesseract):
        for paddle_index, paddle_line in enumerate(paddle):
            similarity = _text_similarity(tess_line.text, paddle_line.text)
            tess_box = _bbox(tess_line)
            paddle_box = _bbox(paddle_line)
            overlap = _overlap_of_smaller(tess_box, paddle_box)
            same_row = _same_row(tess_box, paddle_box)
            if tess_box is None or paddle_box is None:
                plausible_location = tess_box is None and paddle_box is None
                text_match = similarity >= 0.86
            else:
                plausible_location = same_row and (
                    overlap >= 0.15 or _horizontal_gap(tess_box, paddle_box) <= 0.04
                )
                text_match = similarity >= 0.72 and plausible_location
            if text_match or (overlap >= 0.45 and same_row):
                candidates.append(
                    (0.6 * similarity + 0.4 * overlap, tess_index, paddle_index)
                )
    candidates.sort(reverse=True)
    matched_tesseract: set[int] = set()
    matched_paddle: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, tess_index, paddle_index in candidates:
        if tess_index in matched_tesseract or paddle_index in matched_paddle:
            continue
        matched_tesseract.add(tess_index)
        matched_paddle.add(paddle_index)
        pairs.append((tess_index, paddle_index))
    return pairs


def _choose_line(
    tesseract: OCRLine,
    paddle: OCRLine,
    native_regions: list[NativeTextRegion],
) -> tuple[OCRLine, str]:
    tess_support = _native_support(tesseract, native_regions)
    paddle_support = _native_support(paddle, native_regions)
    if tess_support is not None and paddle_support is not None:
        if tess_support >= paddle_support + 0.12:
            return tesseract, "stronger_native_text_support"
        if paddle_support >= tess_support + 0.12:
            return paddle, "stronger_native_text_support"

    tess_noise = _noise_score(tesseract.text)
    paddle_noise = _noise_score(paddle.text)
    if tess_noise + 0.15 < paddle_noise:
        return tesseract, "fewer_suspicious_tokens"
    if paddle_noise + 0.15 < tess_noise:
        return paddle, "fewer_suspicious_tokens"

    same_text = _same_text(tesseract.text, paddle.text)
    if same_text:
        return tesseract, "engine_agreement"
    return paddle, "ambiguous_conflict_defaulted_to_fallback_engine"


def _native_support(
    line: OCRLine,
    regions: list[NativeTextRegion],
) -> float | None:
    line_box = _bbox(line)
    line_tokens = _tokens(line.text)
    if line_box is None or not line_tokens:
        return None
    scores: list[float] = []
    for region in regions:
        region_box = region.bbox
        region_tokens = _tokens(region.text)
        if region_box is None or not region_tokens:
            continue
        if _overlap_of_first(line_box, region_box) < 0.35:
            continue
        supported_tokens = 0
        for token in line_tokens:
            if any(
                token == region_token
                or (
                    len(token) > 2
                    and SequenceMatcher(None, token, region_token).ratio() >= 0.82
                )
                for region_token in region_tokens
            ):
                supported_tokens += 1
        scores.append(supported_tokens / len(line_tokens))
    return max(scores) if scores else None


def _noise_score(text: str) -> float:
    tokens = _TOKEN.findall(text)
    if not tokens:
        return 1.0
    suspicious = sum(bool(_SUSPICIOUS.search(token)) for token in tokens)
    short_alpha = sum(
        len(token) == 1 and token.isalpha() for token in tokens
    )
    return (suspicious + 0.5 * short_alpha) / len(tokens)


def _text_similarity(first: str, second: str) -> float:
    normalized_first = _normalized_text(first)
    normalized_second = _normalized_text(second)
    if not normalized_first or not normalized_second:
        return 0.0
    return SequenceMatcher(None, normalized_first, normalized_second).ratio()


def _same_text(first: str, second: str) -> bool:
    normalize_spacing = lambda value: " ".join(value.casefold().split())
    return normalize_spacing(first) == normalize_spacing(second)


def _normalized_text(text: str) -> str:
    return " ".join(_TOKEN.findall(text.casefold()))


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


def _bbox(line: OCRLine) -> tuple[float, float, float, float] | None:
    if not line.polygon:
        return None
    xs = [point[0] for point in line.polygon]
    ys = [point[1] for point in line.polygon]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _overlap_of_smaller(
    first: tuple[float, float, float, float] | None,
    second: tuple[float, float, float, float] | None,
) -> float:
    if first is None or second is None:
        return 0.0
    intersection = _intersection(first, second)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    smaller_area = min(first_area, second_area)
    return intersection / smaller_area if smaller_area > 0 else 0.0


def _overlap_of_first(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    return _intersection(first, second) / first_area if first_area > 0 else 0.0


def _horizontal_gap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    if first[2] < second[0]:
        return second[0] - first[2]
    if second[2] < first[0]:
        return first[0] - second[2]
    return 0.0


def _intersection(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def _same_row(
    first: tuple[float, float, float, float] | None,
    second: tuple[float, float, float, float] | None,
) -> bool:
    if first is None or second is None:
        return False
    overlap = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    smaller_height = min(first[3] - first[1], second[3] - second[1])
    if smaller_height <= 0:
        return False
    return overlap / smaller_height >= 0.45


def _reading_order(
    line: OCRLine,
    fallback_index: int,
) -> tuple[float, float, int]:
    box = _bbox(line)
    if box is None:
        return 2.0, 2.0, fallback_index
    return box[1], box[0], fallback_index


def _bbox_metadata(line: OCRLine) -> dict[str, Any]:
    box = _bbox(line)
    if box is None:
        return {}
    return {"bbox_normalized": [round(value, 6) for value in box]}
