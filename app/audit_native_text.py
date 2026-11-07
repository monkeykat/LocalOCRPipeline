from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from app.text_quality import assess_native_text


_PAGE_MARKER = re.compile(r"(?m)^=== PAGE (\d+) ===\s*$")


def audit_output_directory(output_dir: Path) -> dict[str, Any]:
    scanned = 0
    missing_text = 0
    flagged: list[dict[str, Any]] = []
    for metadata_path in sorted(output_dir.rglob("*.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict) or metadata.get("extraction_method") != "native":
            continue

        scanned += 1
        text_path = metadata_path.with_suffix(".txt")
        try:
            text = text_path.read_text(encoding="utf-8")
        except OSError:
            missing_text += 1
            flagged.append(
                {
                    "file": metadata_path.relative_to(output_dir).as_posix(),
                    "reason": "missing_text_output",
                    "pages": [],
                }
            )
            continue

        _, separator, body = text.partition("\n\n")
        if not separator:
            body = text
        markers = list(_PAGE_MARKER.finditer(body))
        page_results: list[dict[str, Any]] = []
        if markers:
            for index, marker in enumerate(markers):
                end = markers[index + 1].start() if index + 1 < len(markers) else len(body)
                assessment = assess_native_text(body[marker.end() : end])
                if assessment["ocr_recommended"]:
                    page_results.append({"page": int(marker.group(1)), **assessment})
        else:
            assessment = assess_native_text(body)
            if assessment["ocr_recommended"]:
                page_results.append({"page": None, **assessment})

        if page_results:
            flagged.append(
                {
                    "file": metadata_path.relative_to(output_dir).as_posix(),
                    "reason": "native_text_quality_review",
                    "pages": page_results,
                }
            )

    return {
        "output_dir": str(output_dir),
        "native_documents_scanned": scanned,
        "missing_text_outputs": missing_text,
        "documents_recommended_for_review": len(flagged),
        "documents": flagged,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only scan of native extraction sidecars for pages whose text "
            "looks too sparse or distorted to trust."
        )
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("data/output_fused"),
        help="OCR output directory (default: data/output_fused)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a short report",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        parser.error(f"output directory does not exist: {output_dir}")

    result = audit_output_directory(output_dir)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(
        f"Scanned {result['native_documents_scanned']} native outputs; "
        f"{result['documents_recommended_for_review']} need text-quality review."
    )
    if result["missing_text_outputs"]:
        print(f"Missing text sidecars: {result['missing_text_outputs']}")
    for document in result["documents"]:
        if document["reason"] == "missing_text_output":
            print(f"REVIEW {document['file']}: text sidecar is missing")
            continue
        for page in document["pages"]:
            page_label = f"page {page['page']}" if page["page"] else "document"
            reasons = ", ".join(page["reasons"])
            ratio = page["alphabetic_ratio"]
            print(
                f"REVIEW {document['file']} {page_label}: "
                f"{reasons}; alphabetic ratio {ratio:.3f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
