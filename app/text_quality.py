from __future__ import annotations

import re
from typing import Any


_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)?")
_SUSPICIOUS_WORD_PATTERN = re.compile(
    r"[A-Za-z][~`#$%&+<>;|][A-Za-z]|([A-Za-z])\1\1|"
    r"[A-Z][a-z]+[A-Z][a-z]"
)
_MINIMUM_NATIVE_ALNUM = 20
_MINIMUM_ALNUM_FOR_RATIO_CHECK = 100
_MINIMUM_ALPHABETIC_RATIO = 0.05


def assess_native_text(text: str) -> dict[str, Any]:
    """Flag extractable text layers that are unlikely to be readable text."""
    reasons: list[str] = []
    if not text.strip():
        reasons.append("empty_text")
    if "\ufffd" in text:
        reasons.append("replacement_character")
    if any(
        ord(character) < 32 and character not in "\n\r\t\f"
        for character in text
    ):
        reasons.append("control_character")

    alphanumeric_count = sum(character.isalnum() for character in text)
    alphabetic_count = sum(character.isalpha() for character in text)
    alphabetic_ratio = (
        alphabetic_count / alphanumeric_count if alphanumeric_count else 0.0
    )
    words = _WORD_PATTERN.findall(text)
    if (
        alphanumeric_count < _MINIMUM_NATIVE_ALNUM
        and alphabetic_count < 5
        and len(words) < 2
    ):
        reasons.append("too_little_text")
    elif (
        alphanumeric_count >= _MINIMUM_ALNUM_FOR_RATIO_CHECK
        and alphabetic_ratio < _MINIMUM_ALPHABETIC_RATIO
    ):
        reasons.append("very_low_alphabetic_ratio")

    suspicious_words = sum(
        bool(_SUSPICIOUS_WORD_PATTERN.search(word)) for word in words
    )
    suspicious_characters = sum(
        character in "~`#$%&<>|^" for character in text
    )
    if suspicious_words >= 2 or suspicious_characters >= 2:
        reasons.append("suspicious_character_patterns")

    return {
        "character_count": len(text),
        "alphanumeric_character_count": alphanumeric_count,
        "alphabetic_character_count": alphabetic_count,
        "alphabetic_ratio": round(alphabetic_ratio, 4),
        "ocr_recommended": bool(reasons),
        "reasons": reasons,
    }
