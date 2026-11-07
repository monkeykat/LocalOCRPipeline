from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.paths import ProjectPaths
from app.processor import DocumentProcessor, ProcessedDocument

_processor: DocumentProcessor | None = None


def initialize_worker(
    paths: ProjectPaths,
    workload_type: str | None = None,
    registration_queue: Any | None = None,
) -> None:
    """Create one processor per worker without resetting shared database state."""
    global _processor
    _processor = DocumentProcessor(paths, reset_interrupted=False)
    if workload_type is not None and registration_queue is not None:
        registration_queue.put((workload_type, os.getpid()))


def process_document(source_path: Path) -> ProcessedDocument:
    if _processor is None:
        raise RuntimeError("OCR worker has not been initialized")
    return _processor.process_document(source_path)
