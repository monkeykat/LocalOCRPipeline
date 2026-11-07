from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from app.text_quality import assess_native_text


# Increment this when classification rules or relevant text-quality thresholds change.
CLASSIFIER_VERSION = "1"


@dataclass(frozen=True)
class PreflightClassification:
    workload_type: str
    reason: str


def classify_pdf(path: Path) -> PreflightClassification:
    """Route only high-confidence text PDFs to the native worker pool."""
    try:
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            return PreflightClassification("ocr", "encrypted_pdf")
        if _has_embedded_files(reader):
            return PreflightClassification("ocr", "embedded_files")
        if not reader.pages:
            return PreflightClassification("ocr", "no_pages")

        for page in reader.pages:
            if _page_has_image_resources(page):
                return PreflightClassification("ocr", "image_resources")
            text = page.extract_text() or ""
            assessment = assess_native_text(text)
            if assessment["ocr_recommended"]:
                reasons = assessment["reasons"]
                reason = reasons[0] if reasons else "uncertain_native_text"
                return PreflightClassification("ocr", reason)
        return PreflightClassification("native", "confident_native_text")
    except Exception as error:
        return PreflightClassification("ocr", f"preflight_error:{type(error).__name__}")


def _resolve(value: Any) -> Any:
    getter = getattr(value, "get_object", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return value
    return value


def _has_embedded_files(reader: PdfReader) -> bool:
    try:
        root = _resolve(reader.trailer.get("/Root"))
        names = _resolve(root.get("/Names")) if root is not None else None
        if names is not None and names.get("/EmbeddedFiles") is not None:
            return True
        return bool(root is not None and root.get("/AF"))
    except Exception:
        return True


def _page_has_image_resources(page: Any) -> bool:
    try:
        resources = page.get("/Resources")
        if resources is None and hasattr(page, "get_inherited"):
            resources = page.get_inherited("/Resources")
        return _resources_have_image(resources, set())
    except Exception:
        # A page whose resources cannot be inspected is routed conservatively.
        return True


def _resources_have_image(resources: Any, visited: set[int]) -> bool:
    resources = _resolve(resources)
    if resources is None or id(resources) in visited:
        return False
    visited.add(id(resources))
    if not hasattr(resources, "get"):
        return False
    xobjects = _resolve(resources.get("/XObject"))
    if xobjects is None or not hasattr(xobjects, "values"):
        return False

    for reference in xobjects.values():
        xobject = _resolve(reference)
        if not hasattr(xobject, "get"):
            continue
        subtype = str(xobject.get("/Subtype", ""))
        if subtype == "/Image":
            return True
        if subtype == "/Form" and _resources_have_image(
            xobject.get("/Resources"), visited
        ):
            return True
    return False


def _needs_ocr(text: str) -> bool:
    return bool(assess_native_text(text)["ocr_recommended"])


