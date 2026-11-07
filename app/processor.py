from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from app.database import (
    DocumentStatus,
    ExtractionMethod,
    complete_document,
    connect_database,
    has_completed_hash,
    initialize_database,
    record_error,
    register_document,
    reset_interrupted_documents,
    set_document_status,
    update_page_count,
)
from app.extractor import EmbeddedPDF, PDFExtraction, PdfExtractor
from app.paths import (
    ProjectPaths,
    ensure_data_directories,
    failed_path_for,
    get_project_paths,
    output_path_for,
)
from app.postprocessor import clean_transcript, review_flags_by_page
from app.quality import assess_extraction_quality
from app.text_quality import assess_native_text

_EXTRACTION_PIPELINE_VERSION = 10
_FUSION_ALGORITHM_VERSION = 2


class Extractor(Protocol):
    def extract(
        self,
        source: Path | bytes,
        filename: str | None = None,
    ) -> PDFExtraction: ...


@dataclass(frozen=True)
class ProcessedDocument:
    filename: str
    file_path: str
    status: DocumentStatus
    page_count: int | None = None
    extraction_method: ExtractionMethod | None = None
    skipped: bool = False
    error: str | None = None
    children: tuple[ProcessedDocument, ...] = ()


class DocumentProcessor:
    def __init__(
        self,
        paths: ProjectPaths | None = None,
        extractor: Extractor | None = None,
        *,
        reset_interrupted: bool = True,
    ) -> None:
        self.paths = paths or get_project_paths()
        self.extractor = extractor or PdfExtractor()
        self.ocr_engine = str(
            getattr(self.extractor, "ocr_engine", os.getenv("OCR_ENGINE", "paddleocr"))
        ).strip().lower()
        self.fusion_threshold = (
            float(
                getattr(
                    self.extractor,
                    "fusion_threshold",
                    os.getenv("OCR_FUSION_THRESHOLD", "90"),
                )
            )
            if self.ocr_engine == "fused"
            else None
        )
        self.post_cleanup_enabled = _read_bool_env("OCR_POST_CLEANUP", default=True)
        ensure_data_directories(self.paths)
        if reset_interrupted:
            initialize_database(self.paths)
            with connect_database(self.paths.database_path) as connection:
                reset_interrupted_documents(connection)

    def process_document(self, source_path: Path) -> ProcessedDocument:
        source = source_path.resolve()
        try:
            relative_path = source.relative_to(self.paths.input_dir.resolve())
        except ValueError as error:
            raise ValueError(f"PDF must be inside {self.paths.input_dir}") from error
        if not source.is_file():
            raise FileNotFoundError(source)
        return self._process_content(
            content_source=source,
            filename=source.name,
            database_path=str(source),
            output_relative_path=relative_path,
            failed_path=failed_path_for(source, self.paths),
            parent_context=None,
            ancestor_hashes=frozenset(),
        )

    def record_worker_failure(
        self,
        source_path: Path,
        error_message: str,
        *,
        file_hash: str | None = None,
    ) -> bool:
        """Persist a terminal worker failure after bounded retries."""
        source = source_path.resolve()
        relative_path = source.relative_to(self.paths.input_dir.resolve())
        content_hash = file_hash or _hash_content(source)
        database_path = str(source)
        with connect_database(self.paths.database_path) as connection:
            row = connection.execute(
                """
                SELECT id, status FROM documents
                WHERE file_path = ? AND file_hash = ?
                """,
                (database_path, content_hash),
            ).fetchone()
            if row is None:
                document_id = register_document(
                    connection,
                    filename=source.name,
                    file_path=database_path,
                    file_hash=content_hash,
                )
            else:
                if row["status"] == DocumentStatus.COMPLETE.value:
                    return False
                document_id = int(row["id"])

        self._remove_output_artifacts(relative_path)
        self._fail_document(
            document_id,
            error_message,
            source,
            failed_path_for(source, self.paths),
        )
        return True

    def _process_content(
        self,
        *,
        content_source: Path | bytes,
        filename: str,
        database_path: str,
        output_relative_path: Path,
        failed_path: Path,
        parent_context: dict[str, str | int] | None,
        ancestor_hashes: frozenset[str],
    ) -> ProcessedDocument:
        file_hash = _hash_content(content_source)
        cached_document: sqlite3.Row | None = None
        with connect_database(self.paths.database_path) as connection:
            document_id = register_document(
                connection,
                filename=filename,
                file_path=database_path,
                file_hash=file_hash,
            )
            is_cycle = file_hash in ancestor_hashes
            if is_cycle:
                set_document_status(
                    connection,
                    document_id,
                    DocumentStatus.PROCESSING,
                )
            elif has_completed_hash(connection, file_hash):
                cached_document = connection.execute(
                    """
                    SELECT page_count, extraction_method
                    FROM documents
                    WHERE file_hash = ? AND status = 'COMPLETE'
                    ORDER BY completed_at DESC, id DESC
                    LIMIT 1
                    """,
                    (file_hash,),
                ).fetchone()
                set_document_status(
                    connection,
                    document_id,
                    DocumentStatus.PROCESSING,
                )
            else:
                set_document_status(connection, document_id, DocumentStatus.PROCESSING)

        if file_hash in ancestor_hashes:
            error_message = "Embedded PDF cycle detected; attachment content matches an ancestor"
            self._fail_document(
                document_id,
                error_message,
                content_source,
                failed_path,
            )
            return ProcessedDocument(
                filename=filename,
                file_path=database_path,
                status=DocumentStatus.FAILED,
                error=error_message,
            )

        if cached_document is not None:
            output_path = self._output_path(output_relative_path)
            metadata_path = output_path.with_suffix(".json")
            cached_output = self._find_cached_output(
                file_hash,
                preferred_metadata_path=metadata_path,
            )
            if cached_output is None:
                cached_document = None
            else:
                try:
                    self._publish_cached_output(
                        cached_output=cached_output,
                        output_path=output_path,
                        metadata_path=metadata_path,
                        filename=filename,
                        file_hash=file_hash,
                        parent_context=parent_context,
                    )
                    extraction_method = ExtractionMethod(
                        cached_output["metadata"]["extraction_method"]
                    )
                    page_count = int(cached_output["metadata"]["page_count"])
                    with connect_database(self.paths.database_path) as connection:
                        complete_document(
                            connection,
                            document_id,
                            page_count=page_count,
                            extraction_method=extraction_method,
                        )
                    self._remove_failed_copy(failed_path)
                except Exception as error:
                    self._remove_output_artifacts(output_relative_path)
                    message = f"{type(error).__name__}: {error}"
                    self._fail_document(
                        document_id,
                        message,
                        content_source,
                        failed_path,
                    )
                    return ProcessedDocument(
                        filename=filename,
                        file_path=database_path,
                        status=DocumentStatus.FAILED,
                        error=message,
                    )
                return ProcessedDocument(
                    filename=filename,
                    file_path=database_path,
                    status=DocumentStatus.COMPLETE,
                    page_count=page_count,
                    extraction_method=extraction_method,
                    skipped=True,
                )

        self._remove_output_artifacts(output_relative_path)
        try:
            extraction = self.extractor.extract(content_source, filename=filename)
            if extraction.page_count <= 0:
                raise ValueError("PDF extraction returned a non-positive page count")
            with connect_database(self.paths.database_path) as connection:
                update_page_count(connection, document_id, extraction.page_count)

            output_path = self._output_path(output_relative_path)
            metadata_path = output_path.with_suffix(".json")
            self._publish_outputs(
                output_path=output_path,
                metadata_path=metadata_path,
                filename=filename,
                file_hash=file_hash,
                extraction=extraction,
                parent_context=parent_context,
            )
            with connect_database(self.paths.database_path) as connection:
                complete_document(
                    connection,
                    document_id,
                    page_count=extraction.page_count,
                    extraction_method=extraction.extraction_method,
                )
            self._remove_failed_copy(failed_path)
        except Exception as error:
            self._remove_output_artifacts(output_relative_path)
            message = f"{type(error).__name__}: {error}"
            self._fail_document(document_id, message, content_source, failed_path)
            return ProcessedDocument(
                filename=filename,
                file_path=database_path,
                status=DocumentStatus.FAILED,
                error=message,
            )

        children = tuple(
            self._process_embedded(
                attachment,
                parent_database_path=database_path,
                parent_output_path=output_relative_path,
                parent_hash=file_hash,
                ancestor_hashes=ancestor_hashes | {file_hash},
            )
            for attachment in extraction.embedded_pdfs
        )
        return ProcessedDocument(
            filename=filename,
            file_path=database_path,
            status=DocumentStatus.COMPLETE,
            page_count=extraction.page_count,
            extraction_method=extraction.extraction_method,
            children=children,
        )

    def _process_embedded(
        self,
        attachment: EmbeddedPDF,
        *,
        parent_database_path: str,
        parent_output_path: Path,
        parent_hash: str,
        ancestor_hashes: frozenset[str],
    ) -> ProcessedDocument:
        embedded_relative_path = _embedded_output_path(
            parent_output_path,
            attachment.attachment_index,
            attachment.filename,
        )
        embedded_database_path = (
            f"embedded://{parent_database_path}"
            f"#{attachment.attachment_index}:{attachment.filename}:{parent_hash}"
        )
        failed_relative_path = Path("embedded") / parent_hash / (
            f"{attachment.attachment_index:04d}-{attachment.filename}"
        )
        return self._process_content(
            content_source=attachment.content,
            filename=attachment.filename,
            database_path=embedded_database_path,
            output_relative_path=embedded_relative_path,
            failed_path=self.paths.failed_dir / failed_relative_path,
            parent_context={
                "file_path": parent_database_path,
                "filename": attachment.parent_filename,
                "attachment_index": attachment.attachment_index,
            },
            ancestor_hashes=ancestor_hashes,
        )

    def _output_path(self, relative_path: Path) -> Path:
        if relative_path.parts and relative_path.parts[0] == "@embedded":
            embedded_path = Path(*relative_path.parts[1:])
            return (self.paths.output_dir / embedded_path).with_suffix(".txt")
        return output_path_for(self.paths.input_dir / relative_path, self.paths)

    def _publish_outputs(
        self,
        *,
        output_path: Path,
        metadata_path: Path,
        filename: str,
        file_hash: str,
        extraction: PDFExtraction,
        parent_context: dict[str, str | int] | None,
    ) -> None:
        metadata: dict[str, object] = {
            "filename": filename,
            "file_hash": file_hash,
            "pipeline_version": _EXTRACTION_PIPELINE_VERSION,
            "ocr_engine": extraction.ocr_engine,
            "page_count": extraction.page_count,
            "extraction_method": extraction.extraction_method.value,
            "status": DocumentStatus.COMPLETE.value.lower(),
        }
        review_flags = review_flags_by_page(extraction.ocr_lines)
        metadata["quality_assessment"] = (
            extraction.quality_assessment
            or assess_extraction_quality(extraction, review_flags)
        )
        if extraction.native_text_assessment is not None:
            metadata["native_text_assessment"] = extraction.native_text_assessment
        if extraction.fusion_assessment is not None:
            metadata["fusion_assessment"] = extraction.fusion_assessment
        if self.post_cleanup_enabled:
            metadata["post_cleanup"] = {
                "enabled": True,
                "cleaned_output": _cleaned_output_path(output_path).name,
                "word_corrections_applied": 0,
            }
            if review_flags:
                metadata["ocr_review_flags_by_page"] = review_flags
        if extraction.ocr_pages:
            metadata["ocr_pages"] = list(extraction.ocr_pages)
        if extraction.ocr_time_seconds is not None:
            metadata["ocr_time_seconds"] = round(extraction.ocr_time_seconds, 3)
        if extraction.extraction_time_seconds is not None:
            metadata["extraction_time_seconds"] = round(
                extraction.extraction_time_seconds,
                3,
            )
        if extraction.ocr_confidence:
            metadata["ocr_confidence_by_page"] = {
                str(page_number): round(confidence, 4)
                for page_number, confidence in extraction.ocr_confidence
            }
        if extraction.ocr_lines:
            metadata["ocr_lines_by_page"] = {
                str(page_number): [
                    {
                        "text": line.text,
                        **(
                            {"confidence": round(line.confidence, 4)}
                            if line.confidence is not None
                            else {}
                        ),
                        **(
                            {
                                "polygon_normalized": [
                                    [round(x, 6), round(y, 6)]
                                    for x, y in line.polygon
                                ]
                            }
                            if line.polygon is not None
                            else {}
                        ),
                    }
                    for line in lines
                ]
                for page_number, lines in extraction.ocr_lines
            }
        if extraction.native_regions:
            metadata["native_text_regions_by_page"] = {
                str(page_number): [
                    {
                        "text": region.text,
                        **(
                            {
                                "bbox_normalized": [
                                    round(value, 6) for value in region.bbox
                                ]
                            }
                            if region.bbox is not None
                            else {}
                        ),
                    }
                    for region in regions
                ]
                for page_number, regions in extraction.native_regions
            }
        if parent_context is not None:
            metadata["parent"] = parent_context

        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(metadata_path, json.dumps(metadata, indent=2) + "\n")
        try:
            _atomic_write_text(
                output_path,
                f"DOCUMENT: {filename}\n\n{extraction.text}\n",
            )
            cleaned_path = _cleaned_output_path(output_path)
            if self.post_cleanup_enabled:
                _atomic_write_text(
                    cleaned_path,
                    clean_transcript(f"DOCUMENT: {filename}\n\n{extraction.text}\n"),
                )
            else:
                cleaned_path.unlink(missing_ok=True)
        except Exception:
            metadata_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            _cleaned_output_path(output_path).unlink(missing_ok=True)
            raise

    def _find_cached_output(
        self,
        file_hash: str,
        *,
        preferred_metadata_path: Path | None = None,
    ) -> dict[str, object] | None:
        if preferred_metadata_path is not None:
            cached_output = self._read_cached_output(
                preferred_metadata_path,
                file_hash,
            )
            if cached_output is not None:
                return cached_output

        for metadata_path in self.paths.output_dir.rglob("*.json"):
            if metadata_path == preferred_metadata_path:
                continue
            cached_output = self._read_cached_output(metadata_path, file_hash)
            if cached_output is not None:
                return cached_output
        return None

    def has_reusable_output(
        self,
        file_hash: str,
        metadata_path: Path,
    ) -> bool:
        """Return whether this path has a valid output for the active settings."""
        return self._read_cached_output(metadata_path, file_hash) is not None

    def _read_cached_output(
        self,
        metadata_path: Path,
        file_hash: str,
    ) -> dict[str, object] | None:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(metadata, dict) or metadata.get("file_hash") != file_hash:
            return None
        if metadata.get("pipeline_version") != _EXTRACTION_PIPELINE_VERSION:
            return None
        if metadata.get("ocr_engine", "paddleocr") != self.ocr_engine:
            return None
        if self.ocr_engine == "fused":
            fusion_assessment = metadata.get("fusion_assessment")
            if not isinstance(fusion_assessment, dict):
                return None
            try:
                cached_threshold = float(fusion_assessment["threshold"])
            except (KeyError, TypeError, ValueError):
                return None
            if cached_threshold != self.fusion_threshold:
                return None
            if (
                fusion_assessment.get("paddle_fallback_triggered") is True
                and fusion_assessment.get("fusion_algorithm_version")
                != _FUSION_ALGORITHM_VERSION
            ):
                return None
        text_path = metadata_path.with_suffix(".txt")
        if not text_path.is_file():
            return None
        if (
            metadata.get("extraction_method") == ExtractionMethod.NATIVE.value
            and _cached_native_text_needs_ocr(text_path)
        ):
            return None
        return {"metadata": metadata, "text_path": text_path}

    def _publish_cached_output(
        self,
        *,
        cached_output: dict[str, object],
        output_path: Path,
        metadata_path: Path,
        filename: str,
        file_hash: str,
        parent_context: dict[str, str | int] | None,
    ) -> None:
        cached_text_path = cached_output["text_path"]
        if not isinstance(cached_text_path, Path):
            raise TypeError("Cached output path is invalid")
        existing_text = cached_text_path.read_text(encoding="utf-8")
        _, separator, extracted_text = existing_text.partition("\n\n")
        if not separator:
            raise ValueError(f"Cached text output is malformed: {cached_text_path}")
        cached_metadata = cached_output["metadata"]
        if not isinstance(cached_metadata, dict):
            raise TypeError("Cached output metadata is invalid")
        metadata: dict[str, object] = {
            "filename": filename,
            "file_hash": file_hash,
            "pipeline_version": _EXTRACTION_PIPELINE_VERSION,
            "ocr_engine": self.ocr_engine,
            "page_count": cached_metadata["page_count"],
            "extraction_method": cached_metadata["extraction_method"],
            "status": DocumentStatus.COMPLETE.value.lower(),
        }
        for key in (
            "ocr_pages",
            "ocr_time_seconds",
            "extraction_time_seconds",
            "ocr_confidence_by_page",
            "ocr_lines_by_page",
            "native_text_regions_by_page",
            "native_text_assessment",
            "quality_assessment",
            "fusion_assessment",
        ):
            if key in cached_metadata:
                metadata[key] = cached_metadata[key]
        if self.post_cleanup_enabled:
            metadata["post_cleanup"] = {
                "enabled": True,
                "cleaned_output": _cleaned_output_path(output_path).name,
                "word_corrections_applied": 0,
            }
            if "ocr_review_flags_by_page" in cached_metadata:
                metadata["ocr_review_flags_by_page"] = cached_metadata[
                    "ocr_review_flags_by_page"
                ]
        if parent_context is not None:
            metadata["parent"] = parent_context

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(metadata_path, json.dumps(metadata, indent=2) + "\n")
        try:
            _atomic_write_text(
                output_path,
                f"DOCUMENT: {filename}\n\n{extracted_text}",
            )
            cleaned_path = _cleaned_output_path(output_path)
            if self.post_cleanup_enabled:
                _atomic_write_text(
                    cleaned_path,
                    clean_transcript(f"DOCUMENT: {filename}\n\n{extracted_text}"),
                )
            else:
                cleaned_path.unlink(missing_ok=True)
        except Exception:
            metadata_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            _cleaned_output_path(output_path).unlink(missing_ok=True)
            raise

    def _remove_output_artifacts(self, output_relative_path: Path) -> None:
        output_path = self._output_path(output_relative_path)
        output_path.unlink(missing_ok=True)
        output_path.with_suffix(".json").unlink(missing_ok=True)
        _cleaned_output_path(output_path).unlink(missing_ok=True)

    @staticmethod
    def _remove_failed_copy(failed_path: Path) -> None:
        try:
            failed_path.unlink(missing_ok=True)
        except OSError:
            # Failure copies are secondary artifacts; don't invalidate a
            # document that has already been published and marked complete.
            pass

    def _fail_document(
        self,
        document_id: int,
        error_message: str,
        content_source: Path | bytes,
        failed_path: Path,
    ) -> None:
        failures = [error_message]
        try:
            _publish_failed_source(content_source, failed_path)
        except OSError as copy_error:
            failures.append(
                f"Could not copy failed PDF to {failed_path}: "
                f"{type(copy_error).__name__}: {copy_error}"
            )

        with connect_database(self.paths.database_path) as connection:
            set_document_status(connection, document_id, DocumentStatus.FAILED)
            for failure in failures:
                record_error(connection, document_id, failure)


def _hash_content(source: Path | bytes) -> str:
    digest = hashlib.sha256()
    if isinstance(source, bytes):
        digest.update(source)
    else:
        with source.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _cached_native_text_needs_ocr(text_path: Path) -> bool:
    try:
        text = text_path.read_text(encoding="utf-8")
    except OSError:
        return False
    _, separator, body = text.partition("\n\n")
    if not separator:
        body = text
    pages = re.split(r"(?m)^=== PAGE \d+ ===\s*$", body)
    return any(
        "very_low_alphabetic_ratio" in assess_native_text(page)["reasons"]
        for page in pages
        if page.strip()
    )


def _cleaned_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.cleaned.txt")


def _read_bool_env(name: str, *, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    value = raw_value.strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be a boolean value (true/false, 1/0, yes/no, on/off)"
    )


def _embedded_output_path(
    parent_output_path: Path,
    attachment_index: int,
    filename: str,
) -> Path:
    if parent_output_path.parts and parent_output_path.parts[0] == "@embedded":
        parent_output_path = Path(*parent_output_path.parts[1:])
    parent_relative = parent_output_path.with_suffix("")
    embedded_directory = Path(
        f"{parent_relative.as_posix()}.embedded"
    )
    filename_path = PurePosixPath(filename)
    safe_name = filename_path.name or f"embedded-{attachment_index}.pdf"
    return (
        Path("@embedded")
        / embedded_directory
        / f"{attachment_index:04d}-{safe_name}"
    )


def _atomic_write_text(destination: Path, text: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(text)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _publish_failed_source(source: Path | bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(source, bytes):
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(source)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        shutil.copyfile(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
