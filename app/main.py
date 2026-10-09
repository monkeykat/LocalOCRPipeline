from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from app.database import DocumentStatus, ExtractionMethod
from app.paths import ProjectPaths, get_project_paths
from app.processor import DocumentProcessor, ProcessedDocument


@dataclass(frozen=True)
class BatchSummary:
    discovered: int
    documents: int
    pages: int
    native_pages: int
    ocr_pages: int
    failed: int


def scan_pdfs(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    return sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.relative_to(input_dir).as_posix().casefold(),
    )


def run_batch(
    paths: ProjectPaths | None = None,
    *,
    processor: DocumentProcessor | None = None,
    output: TextIO = sys.stdout,
) -> BatchSummary:
    project_paths = paths or get_project_paths()
    pdf_paths = scan_pdfs(project_paths.input_dir)
    document_processor = processor or DocumentProcessor(project_paths)

    print("Local PDF Extractor", file=output)
    print("===================", file=output)
    print(file=output)
    print(f"Found: {len(pdf_paths)} PDFs", file=output)
    print(file=output)
    print("Processing:", file=output)

    counts = {
        "documents": 0,
        "pages": 0,
        "native_pages": 0,
        "ocr_pages": 0,
        "failed": 0,
    }

    for source_path in pdf_paths:
        relative_path = source_path.relative_to(project_paths.input_dir)
        try:
            result = document_processor.process_document(source_path)
        except Exception as error:
            counts["failed"] += 1
            print(
                f"ERROR: {relative_path.as_posix()}",
                file=output,
            )
            print(f"Reason: {type(error).__name__}: {error}", file=output)
            print("Continuing...", file=output)
            continue
        _report_document(result, output, counts, prefix=relative_path.as_posix())

    print(file=output)
    print("Completed:", file=output)
    print(f"  Documents: {counts['documents']}", file=output)
    print(f"  Pages:     {counts['pages']:,}", file=output)
    print(f"  Native:    {counts['native_pages']:,}", file=output)
    print(f"  OCR:       {counts['ocr_pages']:,}", file=output)
    print(f"  Failed:    {counts['failed']}", file=output)

    return BatchSummary(
        discovered=len(pdf_paths),
        documents=counts["documents"],
        pages=counts["pages"],
        native_pages=counts["native_pages"],
        ocr_pages=counts["ocr_pages"],
        failed=counts["failed"],
    )


def _report_document(
    result: ProcessedDocument,
    output: TextIO,
    counts: dict[str, int],
    *,
    prefix: str = "",
) -> None:
    display_name = f"{prefix} [{result.filename}]" if prefix else result.filename
    if result.status is DocumentStatus.FAILED:
        counts["failed"] += 1
        print(f"ERROR: {display_name}", file=output)
        print(f"Reason: {result.error or 'Unknown processing error'}", file=output)
        print("Continuing...", file=output)
        return

    if not result.skipped:
        counts["documents"] += 1
        if result.page_count is not None:
            counts["pages"] += result.page_count
            if result.extraction_method is ExtractionMethod.NATIVE:
                counts["native_pages"] += result.page_count
            elif result.extraction_method is ExtractionMethod.OCR:
                counts["ocr_pages"] += result.page_count
        method = (
            result.extraction_method.value.upper()
            if result.extraction_method is not None
            else "UNKNOWN"
        )
        page_count = (
            f"{result.page_count:>8} pages"
            if result.page_count is not None
            else " page count unknown"
        )
        print(f"  {display_name:<32}{page_count:>16}   {method}", file=output)
    else:
        print(f"  {display_name:<32} already complete; skipped", file=output)

    for child in result.children:
        _report_document(
            child,
            output,
            counts,
            prefix=f"{display_name} -> attachment",
        )


def main() -> int:
    try:
        summary = run_batch()
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
