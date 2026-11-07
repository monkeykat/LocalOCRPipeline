from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, TextIO
from uuid import uuid4

from app.database import (
    connect_database,
    DocumentStatus,
    ExtractionMethod,
    initialize_database,
    register_document,
    reset_interrupted_documents,
    get_classification,
    save_classification,
)
from app.classifier import CLASSIFIER_VERSION, PreflightClassification, classify_pdf
from app.monitor import ResourceMonitor
from app.paths import ProjectPaths, get_project_paths, output_path_for
from app.processor import DocumentProcessor, ProcessedDocument
from app.worker import initialize_worker, process_document as process_in_worker


_MAX_DOCUMENT_RETRIES = 1
_MAX_POOL_RESTARTS_WITHOUT_SUCCESS = 5


@dataclass(frozen=True)
class BatchSummary:
    discovered: int
    documents: int
    pages: int
    native_pages: int
    ocr_pages: int
    failed: int
    unsubmitted: int = 0


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


def _inventory_uncached_documents(
    paths: ProjectPaths,
    discovered_paths: list[Path],
    cache_checker: DocumentProcessor,
    output: TextIO,
) -> tuple[list[Path], int, dict[Path, str]]:
    registered: list[tuple[Path, str, str]] = []
    unhashable: list[Path] = []
    print(
        f"Indexing and hashing {len(discovered_paths)} PDFs into SQLite...",
        file=output,
        flush=True,
    )
    with connect_database(paths.database_path) as connection:
        for index, source_path in enumerate(discovered_paths, start=1):
            try:
                file_hash = _hash_input_file(source_path)
            except OSError as error:
                print(
                    f"WARNING: could not hash {source_path}: {error}; "
                    "leaving it in the work queue",
                    file=output,
                    flush=True,
                )
                unhashable.append(source_path)
                continue

            document_id = register_document(
                connection,
                filename=source_path.name,
                file_path=str(source_path.resolve()),
                file_hash=file_hash,
            )
            row = connection.execute(
                "SELECT status FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Document ID {document_id} disappeared")
            registered.append((source_path, file_hash, str(row["status"])))
            if index % 500 == 0:
                connection.commit()
                print(
                    f"  Indexed and registered {index}/{len(discovered_paths)} PDFs",
                    file=output,
                    flush=True,
                )

    pending: list[Path] = []
    pending_hashes: dict[Path, str] = {}
    skipped = 0
    for source_path, file_hash, status in registered:
        metadata_path = output_path_for(source_path, paths).with_suffix(".json")
        if status == DocumentStatus.COMPLETE.value and cache_checker.has_reusable_output(
            file_hash,
            metadata_path,
        ):
            skipped += 1
        else:
            pending.append(source_path)
            pending_hashes[source_path] = file_hash
    pending.extend(unhashable)
    return pending, skipped, pending_hashes


def _hash_input_file(source_path: Path) -> str:
    digest = hashlib.sha256()
    with source_path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_batch(
    paths: ProjectPaths | None = None,
    *,
    processor: DocumentProcessor | None = None,
    output: TextIO = sys.stdout,
) -> BatchSummary:
    project_paths = paths or get_project_paths()
    discovered_paths = scan_pdfs(project_paths.input_dir)
    inventory_started = time.monotonic()
    cached_skipped = 0
    pending_hashes: dict[Path, str] = {}
    cache_checker: DocumentProcessor | None = None
    if processor is None:
        initialize_database(project_paths)
        with connect_database(project_paths.database_path) as connection:
            reset_interrupted_documents(connection)
        cache_checker = DocumentProcessor(
            project_paths,
            reset_interrupted=False,
        )
        pdf_paths, cached_skipped, pending_hashes = _inventory_uncached_documents(
            project_paths,
            discovered_paths,
            cache_checker,
            output,
        )
    else:
        pdf_paths = discovered_paths
    inventory_duration = time.monotonic() - inventory_started
    requested_workers = _read_worker_count("OCR_WORKERS", 2)
    separate_pools = processor is None and _separate_pool_settings_present()
    native_workers_requested = _read_worker_count(
        "OCR_NATIVE_WORKERS", requested_workers
    ) if separate_pools else 0
    ocr_workers_requested = _read_worker_count(
        "OCR_OCR_WORKERS", requested_workers
    ) if separate_pools else 0
    native_paths: list[Path] = []
    ocr_paths: list[Path] = []
    classification_reasons: Counter[str] = Counter()
    classifications_reused = 0
    classifications_computed = 0
    preflight_started = time.monotonic()
    if separate_pools:
        print(
            f"Classifying {len(pdf_paths)} incomplete PDFs for worker pools...",
            file=output,
            flush=True,
        )
        with connect_database(project_paths.database_path) as connection:
            for index, source_path in enumerate(pdf_paths, start=1):
                file_hash = pending_hashes.get(source_path)
                saved = (
                    get_classification(
                        connection,
                        file_hash=file_hash,
                        classifier_version=CLASSIFIER_VERSION,
                    )
                    if file_hash is not None
                    else None
                )
                if saved is not None:
                    classification = PreflightClassification(*saved)
                    classifications_reused += 1
                else:
                    classification = classify_pdf(source_path)
                    if file_hash is not None:
                        save_classification(
                            connection,
                            file_hash=file_hash,
                            classifier_version=CLASSIFIER_VERSION,
                            workload_type=classification.workload_type,
                            reason=classification.reason,
                        )
                        connection.commit()
                    classifications_computed += 1
                classification_reasons[classification.reason] += 1
                if classification.workload_type == "native":
                    native_paths.append(source_path)
                else:
                    ocr_paths.append(source_path)
                if index % 100 == 0:
                    print(
                        f"  {index}/{len(pdf_paths)} classified "
                        f"({len(native_paths)} native, {len(ocr_paths)} OCR/uncertain; "
                        f"{classifications_reused} reused)",
                        file=output,
                        flush=True,
                    )
        preflight_duration = time.monotonic() - preflight_started
        native_workers_used = min(native_workers_requested, len(native_paths))
        ocr_workers_used = min(ocr_workers_requested, len(ocr_paths))
        worker_count = native_workers_used + ocr_workers_used
    else:
        preflight_duration = 0.0
        native_workers_used = 0
        ocr_workers_used = 0
        worker_count = min(requested_workers, len(pdf_paths))
        if processor is not None and pdf_paths:
            # Keep injected processors (for example, test doubles) in this process.
            worker_count = 1
    document_processor = processor
    if document_processor is None and worker_count <= 1 and not separate_pools:
        document_processor = cache_checker or DocumentProcessor(project_paths)

    print("Local PDF Extractor", file=output)
    print("===================", file=output)
    print(file=output)
    print(f"Found: {len(discovered_paths)} PDFs", file=output)
    if processor is None:
        print(
            f"Already complete: {cached_skipped}; queued for work: {len(pdf_paths)}",
            file=output,
        )
    if separate_pools:
        print(
            f"Workers: {worker_count} total "
            f"({native_workers_used} native, {ocr_workers_used} OCR)",
            file=output,
        )
        print(
            f"Preflight routing: {len(native_paths)} native, {len(ocr_paths)} OCR/uncertain",
            file=output,
        )
        if ocr_paths:
            print(
                f"OCR ETA: based on {len(ocr_paths)} queued OCR/uncertain documents; "
                "estimate appears after the first document completes",
                file=output,
                flush=True,
            )
        print(
            "Preflight classification cache: "
            f"{classifications_reused} reused, {classifications_computed} computed",
            file=output,
        )
    else:
        print(f"Workers: {worker_count}", file=output)
    print(file=output)
    print("Processing:", file=output)

    counts = {
        "documents": 0,
        "pages": 0,
        "native_pages": 0,
        "ocr_pages": 0,
        "failed": 0,
        "preflight_native_to_ocr": 0,
        "pool_jobs_unsubmitted": 0,
        "document_retries": 0,
        "pool_restarts": 0,
        "skipped": 0,
    }
    scheduler_stats: dict[str, object] = {
        "pool_worker_limits_final": {},
        "worker_pools_disabled": [],
    }
    ocr_eta_state: dict[str, object] = {
        "ocr_documents_total": len(ocr_paths) if separate_pools else None,
        "ocr_documents_completed": 0 if separate_pools else None,
        "ocr_avg_document_seconds": None,
        "ocr_estimated_remaining_seconds": None,
        "ocr_estimated_completion_utc": None,
    }

    monitor = ResourceMonitor(sample_interval_seconds=1.0)
    monitor.enable_workload_tracking(worker_count == 1 and not separate_pools)
    pool_registration_queue = get_context("spawn").Queue() if separate_pools else None
    if pool_registration_queue is not None:
        monitor.enable_process_pool_tracking(pool_registration_queue)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_id = f"{run_id}-{uuid4().hex[:8]}"
    engine = os.getenv("OCR_ENGINE", "paddleocr").strip().lower()
    metrics_path = _run_metrics_path(project_paths, engine, run_id)
    run_metadata: dict[str, object] = {
        "run_id": run_id,
        "engine": engine,
        "input_dir": str(project_paths.input_dir),
        "output_dir": str(project_paths.output_dir),
        "workers_requested": (
            native_workers_requested + ocr_workers_requested
            if separate_pools
            else requested_workers
        ),
        "shared_pool_workers_requested": (
            None if separate_pools else requested_workers
        ),
        "workers_used": worker_count,
        "documents_discovered": len(discovered_paths),
        "documents_queued": len(pdf_paths),
        "documents_skipped_cached": cached_skipped,
        "worker_mode": (
            "separate_pools"
            if separate_pools
            else "serial" if worker_count <= 1 else "shared_pool"
        ),
        "native_workers_requested": native_workers_requested if separate_pools else None,
        "native_workers_used": native_workers_used if separate_pools else None,
        "ocr_workers_requested": ocr_workers_requested if separate_pools else None,
        "ocr_workers_used": ocr_workers_used if separate_pools else None,
        "preflight_duration_seconds": round(preflight_duration, 3),
        "inventory_duration_seconds": round(inventory_duration, 3),
        "preflight_native_documents": len(native_paths) if separate_pools else None,
        "preflight_ocr_documents": len(ocr_paths) if separate_pools else None,
        "preflight_reasons": dict(classification_reasons) if separate_pools else None,
        "preflight_classifications_reused": classifications_reused if separate_pools else None,
        "preflight_classifications_computed": classifications_computed if separate_pools else None,
        "classifier_version": CLASSIFIER_VERSION if separate_pools else None,
    }
    monitor.start()
    checkpointer = _RunMetricsCheckpointer(
        monitor,
        metrics_path,
        run_metadata,
        progress_provider=lambda: {
            "documents_processed": counts["documents"],
            "documents_failed": counts["failed"],
            "documents_skipped_cached": cached_skipped + counts["skipped"],
            "preflight_native_to_ocr": counts["preflight_native_to_ocr"],
            "pool_jobs_unsubmitted": counts["pool_jobs_unsubmitted"],
            "document_retries": counts["document_retries"],
            "pool_restarts": counts["pool_restarts"],
            **ocr_eta_state,
        },
        interval_seconds=600.0,
    )
    checkpointer.start()
    try:
        displayed_path = metrics_path.relative_to(project_paths.project_root)
    except ValueError:
        displayed_path = metrics_path
    print(
        f"Live run metrics: {displayed_path} (updated every 10 minutes)",
        file=output,
    )
    run_status = "completed"
    try:
        if separate_pools:
            scheduler_stats.update(_run_process_pools(
                workloads={"native": native_paths, "ocr": ocr_paths},
                capacities={"native": native_workers_used, "ocr": ocr_workers_used},
                initargs_for=lambda workload: (
                    project_paths,
                    workload,
                    pool_registration_queue,
                ),
                input_dir=project_paths.input_dir,
                output=output,
                counts=counts,
                cache_checker=cache_checker,
                pending_hashes=pending_hashes,
                ocr_eta_state=ocr_eta_state,
                refresh_metrics=checkpointer.refresh,
                on_result=lambda result, source_path, workload: _handle_pool_result(
                    result,
                    source_path,
                    workload,
                    project_paths,
                    pending_hashes,
                    output,
                    counts,
                ),
            ))
        elif worker_count <= 1:
            assert document_processor is not None
            for source_path in pdf_paths:
                relative_path = source_path.relative_to(project_paths.input_dir)
                workload_token = monitor.begin_workload()
                result: ProcessedDocument | None = None
                last_error: Exception | None = None
                for attempt in range(_MAX_DOCUMENT_RETRIES + 1):
                    try:
                        result = document_processor.process_document(source_path)
                        last_error = None
                    except Exception as error:
                        last_error = error
                    if last_error is None and result is not None and result.status is not DocumentStatus.FAILED:
                        break
                    if attempt < _MAX_DOCUMENT_RETRIES:
                        counts["document_retries"] += 1
                        print(
                            f"WARNING: retrying {relative_path.as_posix()} "
                            f"({attempt + 1}/{_MAX_DOCUMENT_RETRIES})",
                            file=output,
                            flush=True,
                        )
                if last_error is not None:
                    monitor.finish_workload(workload_token, "unclassified")
                    _report_error(relative_path, last_error, output, counts)
                    continue
                assert result is not None
                workload_type, page_count, skipped = _workload_profile(result)
                monitor.finish_workload(
                    workload_token,
                    workload_type,
                    pages=page_count,
                    skipped=skipped,
                )
                _report_document(
                    result,
                    output,
                    counts,
                    prefix=relative_path.as_posix(),
                )
        else:
            scheduler_stats.update(_run_process_pools(
                workloads={"shared": pdf_paths},
                capacities={"shared": worker_count},
                initargs_for=lambda _workload: (project_paths,),
                input_dir=project_paths.input_dir,
                output=output,
                counts=counts,
                cache_checker=cache_checker,
                pending_hashes=pending_hashes,
                ocr_eta_state=ocr_eta_state,
                refresh_metrics=checkpointer.refresh,
                on_result=lambda result, source_path, _workload: _report_document(
                    result,
                    output,
                    counts,
                    prefix=source_path.relative_to(project_paths.input_dir).as_posix(),
                ),
            ))
        if counts["failed"] or counts["pool_jobs_unsubmitted"]:
            run_status = "completed_with_errors"
    except BaseException as error:
        run_status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        raise
    finally:
        checkpointer.stop()
        run_metrics = monitor.stop()
        if pool_registration_queue is not None:
            pool_registration_queue.close()
        run_metrics.update(
            {
                **run_metadata,
                "documents_processed": counts["documents"],
                "documents_failed": counts["failed"],
                "documents_skipped_cached": cached_skipped + counts["skipped"],
                "preflight_native_to_ocr": counts["preflight_native_to_ocr"],
                "pool_jobs_unsubmitted": counts["pool_jobs_unsubmitted"],
                "document_retries": counts["document_retries"],
                "pool_restarts": counts["pool_restarts"],
                **ocr_eta_state,
                **scheduler_stats,
                "run_status": run_status,
                "checkpoint_interval_seconds": checkpointer.interval_seconds,
            }
        )
        try:
            _write_run_metrics(metrics_path, run_metrics)
            print(f"Final run metrics: {displayed_path}", file=output)
        except OSError as error:
            print(f"WARNING: Could not save run metrics: {error}", file=output)

    print(file=output)
    print(
        "Completed with errors:"
        if counts["failed"] or counts["pool_jobs_unsubmitted"]
        else "Completed:",
        file=output,
    )
    print(f"  Documents: {counts['documents']}", file=output)
    print(f"  Pages:     {counts['pages']:,}", file=output)
    print(f"  Native:    {counts['native_pages']:,}", file=output)
    print(f"  OCR:       {counts['ocr_pages']:,}", file=output)
    print(f"  Failed:    {counts['failed']}", file=output)
    print(f"  Retried:   {counts['document_retries']}", file=output)
    print(f"  Restarts:  {counts['pool_restarts']}", file=output)
    print(f"  Pending:   {counts['pool_jobs_unsubmitted']}", file=output)
    print(
        f"  Cached:    {cached_skipped + counts['skipped']:,} skipped",
        file=output,
    )

    return BatchSummary(
        discovered=len(discovered_paths),
        documents=counts["documents"],
        pages=counts["pages"],
        native_pages=counts["native_pages"],
        ocr_pages=counts["ocr_pages"],
        failed=counts["failed"],
        unsubmitted=counts["pool_jobs_unsubmitted"],
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
        counts["skipped"] += 1
        print(f"  {display_name:<32} already complete; skipped", file=output)

    for child in result.children:
        _report_document(
            child,
            output,
            counts,
            prefix=f"{display_name} -> attachment",
        )


def _workload_profile(result: ProcessedDocument) -> tuple[str, int, bool]:
    """Classify a serial top-level document by whether any page used OCR."""
    methods: list[ExtractionMethod] = []
    pages = 0

    def collect(document: ProcessedDocument) -> None:
        nonlocal pages
        if document.skipped:
            return
        if document.extraction_method is not None:
            methods.append(document.extraction_method)
        pages += document.page_count or 0
        for child in document.children:
            collect(child)

    collect(result)
    if ExtractionMethod.OCR in methods:
        return "ocr", pages, result.skipped
    if ExtractionMethod.NATIVE in methods:
        return "native", pages, result.skipped
    return "unclassified", pages, result.skipped


def _report_error(
    relative_path: Path,
    error: Exception,
    output: TextIO,
    counts: dict[str, int],
) -> None:
    counts["failed"] += 1
    print(f"ERROR: {relative_path.as_posix()}", file=output)
    print(f"Reason: {type(error).__name__}: {error}", file=output)
    print("Continuing...", file=output)


def _handle_pool_result(
    result: ProcessedDocument,
    source_path: Path,
    predicted_type: str,
    paths: ProjectPaths,
    pending_hashes: dict[Path, str],
    output: TextIO,
    counts: dict[str, int],
) -> None:
    relative_path = source_path.relative_to(paths.input_dir)
    actual_type, _, _ = _workload_profile(result)
    if predicted_type == "native" and actual_type == "ocr":
        counts["preflight_native_to_ocr"] += 1
        file_hash = pending_hashes.get(source_path)
        if file_hash is not None:
            with connect_database(paths.database_path) as connection:
                save_classification(
                    connection,
                    file_hash=file_hash,
                    classifier_version=CLASSIFIER_VERSION,
                    workload_type="ocr",
                    reason="observed_ocr_fallback",
                )
        print(
            f"WARNING: {relative_path.as_posix()} was preflighted as native but "
            "the extractor used OCR; review the preflight classifier before "
            "raising the native cap.",
            file=output,
            flush=True,
        )
    _report_document(
        result,
        output,
        counts,
        prefix=relative_path.as_posix(),
    )


def _run_process_pools(
    *,
    workloads: dict[str, list[Path]],
    capacities: dict[str, int],
    initargs_for: Callable[[str], tuple[object, ...]],
    input_dir: Path,
    output: TextIO,
    counts: dict[str, int],
    cache_checker: DocumentProcessor | None,
    pending_hashes: dict[Path, str],
    ocr_eta_state: dict[str, object],
    refresh_metrics: Callable[[], None],
    on_result: Callable[[ProcessedDocument, Path, str], None],
) -> dict[str, object]:
    """Run isolated pools and recover from worker crashes without losing the queue."""
    queues = {name: deque(paths) for name, paths in workloads.items()}
    limits = dict(capacities)
    active = {name: 0 for name in workloads}
    retries: dict[tuple[str, Path], int] = {}
    crash_streak = {name: 0 for name in workloads}
    disabled: set[str] = set()
    executors: dict[str, ProcessPoolExecutor] = {}
    futures: dict[object, tuple[Path, str, float]] = {}
    elapsed_by_document: dict[tuple[str, Path], float] = {}
    ocr_duration_samples: list[float] = []
    context = get_context("spawn")

    def record_completion(workload: str, source_path: Path) -> None:
        if workload != "ocr" or ocr_eta_state.get("ocr_documents_total") is None:
            return
        completed_count = int(ocr_eta_state.get("ocr_documents_completed") or 0) + 1
        ocr_eta_state["ocr_documents_completed"] = completed_count
        sample = elapsed_by_document.pop((workload, source_path), 0.0)
        if sample > 0:
            ocr_duration_samples.append(sample)
        average = (
            sum(ocr_duration_samples) / len(ocr_duration_samples)
            if ocr_duration_samples
            else None
        )
        total = int(ocr_eta_state["ocr_documents_total"] or 0)
        remaining = max(0, total - completed_count)
        workers = max(1, limits.get(workload, 1))
        estimate = remaining * average / workers if average is not None else None
        completion_time = (
            datetime.now(timezone.utc) + timedelta(seconds=estimate)
            if estimate is not None
            else None
        )
        ocr_eta_state.update(
            {
                "ocr_avg_document_seconds": round(average, 3) if average is not None else None,
                "ocr_estimated_remaining_seconds": round(estimate, 3) if estimate is not None else None,
                "ocr_estimated_completion_utc": (
                    completion_time.isoformat(timespec="seconds")
                    if completion_time is not None
                    else None
                ),
            }
        )
        if average is None:
            detail = "ETA available after the first OCR document completes"
        else:
            detail = (
                f"average {_format_duration(average)} per document; "
                f"estimated remaining {_format_duration(estimate or 0)} "
                f"({completion_time.strftime('%Y-%m-%d %H:%M:%S UTC')})"
            )
        print(
            f"OCR progress: {completed_count}/{total} documents; {detail}",
            file=output,
            flush=True,
        )
        refresh_metrics()

    def accumulate_elapsed(workload: str, source_path: Path, elapsed: float) -> None:
        key = (workload, source_path)
        elapsed_by_document[key] = elapsed_by_document.get(key, 0.0) + max(0.0, elapsed)

    def start_executor(workload: str) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=limits[workload],
            mp_context=context,
            initializer=initialize_worker,
            initargs=initargs_for(workload),
        )

    def retry_or_fail(
        workload: str,
        source_path: Path,
        error: Exception,
        *,
        elapsed_seconds: float = 0.0,
    ) -> None:
        key = (workload, source_path)
        accumulate_elapsed(workload, source_path, elapsed_seconds)
        retry_count = retries.get(key, 0)
        if retry_count < _MAX_DOCUMENT_RETRIES:
            retries[key] = retry_count + 1
            counts["document_retries"] += 1
            queues[workload].appendleft(source_path)
            relative = source_path.relative_to(input_dir).as_posix()
            print(
                f"WARNING: {workload} worker failed on {relative}; retrying "
                f"({retry_count + 1}/{_MAX_DOCUMENT_RETRIES}): "
                f"{type(error).__name__}: {error}",
                file=output,
                flush=True,
            )
            return

        if cache_checker is not None:
            try:
                cache_checker.record_worker_failure(
                    source_path,
                    f"Worker failed after {_MAX_DOCUMENT_RETRIES} retry: "
                    f"{type(error).__name__}: {error}",
                    file_hash=pending_hashes.get(source_path),
                )
            except Exception as persist_error:
                print(
                    f"WARNING: could not persist terminal worker failure for "
                    f"{source_path}: {persist_error}",
                    file=output,
                    flush=True,
                )
        _report_error(source_path.relative_to(input_dir), error, output, counts)
        record_completion(workload, source_path)

    def submit_available(workload: str) -> str | None:
        if workload in disabled:
            return None
        executor = executors.get(workload)
        if executor is None:
            return None
        while active[workload] < limits[workload] and queues[workload]:
            source_path = queues[workload].popleft()
            try:
                future = executor.submit(process_in_worker, source_path)
            except BrokenProcessPool as error:
                return handle_broken_pool(workload, error, source_path)
            futures[future] = (source_path, workload, time.monotonic())
            active[workload] += 1
        return None

    def handle_broken_pool(
        workload: str,
        error: Exception,
        not_submitted: Path | None = None,
        failed_assignment: tuple[Path, float] | None = None,
    ) -> str | None:
        affected: list[tuple[Path, float]] = []
        now = time.monotonic()
        for future, (source_path, future_workload, submitted_at) in list(futures.items()):
            if future_workload == workload:
                futures.pop(future, None)
                future.cancel()
                affected.append((source_path, now - submitted_at))
        active[workload] = 0
        if failed_assignment is not None:
            affected.append(failed_assignment)
        if not_submitted is not None:
            affected.append((not_submitted, 0.0))
        for source_path, elapsed in affected:
            retry_or_fail(workload, source_path, error, elapsed_seconds=elapsed)

        executor = executors.pop(workload, None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        counts["pool_restarts"] += 1
        crash_streak[workload] += 1
        if crash_streak[workload] >= _MAX_POOL_RESTARTS_WITHOUT_SUCCESS:
            disabled.add(workload)
            remaining = len(queues[workload])
            counts["pool_jobs_unsubmitted"] += remaining
            if workload == "ocr":
                ocr_eta_state["ocr_estimated_remaining_seconds"] = None
                ocr_eta_state["ocr_estimated_completion_utc"] = None
            print(
                f"ERROR: {workload} pool crashed {crash_streak[workload]} times "
                f"without a successful document. Pausing this pool; {remaining} "
                "PDFs remain pending for the next run.",
                file=output,
                flush=True,
            )
            queues[workload].clear()
            return None

        print(
            f"WARNING: restarting {workload} worker pool with the same "
            f"{limits[workload]} worker(s) after process failure "
            f"({crash_streak[workload]}/{_MAX_POOL_RESTARTS_WITHOUT_SUCCESS}).",
            file=output,
            flush=True,
        )
        time.sleep(0.5)
        executors[workload] = start_executor(workload)
        return None

    try:
        for workload, limit in limits.items():
            if limit > 0:
                executors[workload] = start_executor(workload)
        for workload in workloads:
            submit_available(workload)

        while futures or any(queues[name] for name in workloads if name not in disabled):
            if not futures:
                for workload in workloads:
                    submit_available(workload)
                if not futures:
                    break
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                assignment = futures.pop(future, None)
                if assignment is None:
                    continue
                source_path, workload, submitted_at = assignment
                elapsed = time.monotonic() - submitted_at
                active[workload] -= 1
                try:
                    result = future.result()
                except BrokenProcessPool as error:
                    handle_broken_pool(
                        workload,
                        error,
                        failed_assignment=(source_path, elapsed),
                    )
                    continue
                except Exception as error:
                    retry_or_fail(
                        workload,
                        source_path,
                        error,
                        elapsed_seconds=elapsed,
                    )
                else:
                    accumulate_elapsed(workload, source_path, elapsed)
                    crash_streak[workload] = 0
                    if result.status is DocumentStatus.FAILED:
                        key = (workload, source_path)
                        retry_count = retries.get(key, 0)
                        if retry_count < _MAX_DOCUMENT_RETRIES:
                            retries[key] = retry_count + 1
                            counts["document_retries"] += 1
                            queues[workload].appendleft(source_path)
                            print(
                                f"WARNING: document processing failed for "
                                f"{source_path.relative_to(input_dir)}; retrying "
                                f"({retry_count + 1}/{_MAX_DOCUMENT_RETRIES}): "
                                f"{result.error or 'unknown error'}",
                                file=output,
                                flush=True,
                            )
                        else:
                            on_result(result, source_path, workload)
                            record_completion(workload, source_path)
                    else:
                        on_result(result, source_path, workload)
                        record_completion(workload, source_path)

                submit_available(workload)
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)

    return {
        "pool_worker_limits_final": limits,
        "worker_pools_disabled": sorted(disabled),
    }


def _format_duration(seconds: float) -> str:
    remaining = max(0, int(round(seconds)))
    days, remaining = divmod(remaining, 24 * 60 * 60)
    hours, remaining = divmod(remaining, 60 * 60)
    minutes, secs = divmod(remaining, 60)
    parts = []
    for value, unit in ((days, "d"), (hours, "h"), (minutes, "m"), (secs, "s")):
        if value:
            parts.append(f"{value}{unit}")
        if len(parts) == 2:
            break
    return " ".join(parts) if parts else "0s"


def _read_worker_count(variable: str = "OCR_WORKERS", default: int = 2) -> int:
    raw_value = os.getenv(variable, str(default)).strip() or str(default)
    try:
        worker_count = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{variable} must be a positive integer") from error
    if worker_count < 1:
        raise ValueError(f"{variable} must be a positive integer")
    return worker_count


def _separate_pool_settings_present() -> bool:
    return any(
        os.getenv(variable, "").strip()
        for variable in ("OCR_NATIVE_WORKERS", "OCR_OCR_WORKERS")
    )


def _run_metrics_path(paths: ProjectPaths, engine: str, run_id: str) -> Path:
    metrics_dir = paths.data_dir / "run_metrics"
    safe_engine = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in engine
    ) or "ocr"
    return metrics_dir / f"{safe_engine}-{run_id}.json"


def _write_run_metrics(metrics_path: Path, metrics: dict[str, object]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=metrics_path.parent,
            prefix=f".{metrics_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, metrics_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class _RunMetricsCheckpointer:
    def __init__(
        self,
        monitor: ResourceMonitor,
        metrics_path: Path,
        metadata: dict[str, object],
        *,
        progress_provider: Callable[[], dict[str, object]],
        interval_seconds: float,
    ) -> None:
        self.monitor = monitor
        self.metrics_path = metrics_path
        self.metadata = metadata
        self.progress_provider = progress_provider
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._write_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_error: Exception | None = None

    def start(self) -> None:
        self._write_checkpoint()
        self._thread = threading.Thread(
            target=self._write_periodically,
            name="ocr-run-metrics-writer",
            daemon=True,
        )
        self._thread.start()

    def _write_periodically(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._write_checkpoint()

    def _write_checkpoint(self) -> None:
        with self._write_lock:
            try:
                metrics = self.monitor.snapshot()
                metrics.update(self.metadata)
                metrics.update(self.progress_provider())
                metrics.update(
                    {
                        "run_status": "running",
                        "checkpoint_interval_seconds": self.interval_seconds,
                    }
                )
                _write_run_metrics(self.metrics_path, metrics)
                self._last_error = None
            except Exception as error:
                self._last_error = error

    def refresh(self) -> None:
        """Write a checkpoint immediately after meaningful progress."""
        self._write_checkpoint()

    def stop(self) -> Exception | None:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join()
            self._thread = None
        return self._last_error


def main() -> int:
    try:
        summary = run_batch()
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 1 if summary.failed or summary.unsubmitted else 0


if __name__ == "__main__":
    raise SystemExit(main())
