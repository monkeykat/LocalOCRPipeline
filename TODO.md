# Local PDF Text Extraction & OCR Pipeline — TODO

Implement in order. The target is a fully local Dockerized Python POC on the HP ZGX Nano that produces page-preserved plain text and tracks processing; do not add features listed as out of scope.

## 1. Establish the project and runtime

- [x] Create the documented project layout: `app/`, `data/input/`, `data/output/`, and `data/failed/`.
- [x] Add `requirements.txt` with pinned Docling, PaddleOCR, PaddlePaddle, and NumPy versions.
- [x] Add a `Dockerfile` and `docker-compose.yml` for local runtime, persistent data/model mounts, and a processing service with networking disabled.
- [x] Configure PaddleOCR model prefetching through the `prepare-models` Compose service. Docling's native PDF parser and renderer do not use learned model weights.
- [x] Verify dependency installation and PaddleOCR model preparation in the target container, then process the representative corpus with container networking disabled. The final ARM64 images built successfully; model preparation succeeded and the offline batch completed.
- [x] Verify the container on the target HP ZGX Nano and record runtime settings. Host: ARM64, 20 CPU cores, NVIDIA GB10, 121.6 GiB visible RAM. PaddleOCR ran on CPU with 10 threads; Docker image size was 1.45 GB.
- [x] Document the local setup, model preparation, run command, data mounts, and offline operation in `README.md`.

## 2. Define processing and storage foundations

- [x] Implement `app/database.py` to initialize SQLite and create the specified `documents` and `errors` tables.
- [x] Use the documented `documents` fields (`id`, `filename`, `file_path`, `file_hash`, `page_count`, `extraction_method`, `status`, `created_at`, `completed_at`) and `errors` fields (`id`, `document_id`, `error`, `created_at`); include `file_path` as required by the registration flow and link each error to its document.
- [x] Restrict SQLite to processing/status tracking; use no other database.
- [x] Use the `PENDING`, `PROCESSING`, `COMPLETE`, and `FAILED` statuses, with an explicit completion operation and validated state values.
- [x] Add database operations for registering documents, updating status and extraction metadata, recording errors, and checking whether a matching file hash has already completed successfully.
- [x] Define and test stable paths for input, output, failed files, and SQLite data, including nested input directories.

## 3. Build the extraction layer

- [x] Implement `app/extractor.py` using Docling as the document-processing layer for PDF parsing, document structure, page identification, native text extraction, and coordination with OCR where appropriate; avoid building a custom PDF parser.
- [x] Attempt normal Docling text extraction to determine whether the document has usable text throughout. Use native extraction only for text-based documents; if any of the document is scanned/image-based or otherwise lacks usable text, use PaddleOCR for the document. Keep this a document-level decision rather than adding page-by-page classification.
- [x] Implement the OCR path: render PDF pages as images, recognize text with PaddleOCR, and retain each page's original page number.
- [x] Format both extraction paths consistently as plain text, with `=== PAGE n ===` separators for every original page (including pages with no recognized text) and page boundaries/numbering preserved.
- [x] Handle PDFs that mix usable text and scanned/image-based content according to the document-level decision: run OCR for the document rather than returning incomplete native-only text.
- [x] Discover and extract embedded PDF attachments with pypdf; return their bytes, attachment index, filename, and parent context for registration and processing by the processor.
- [x] Return the extracted text, page count, and extraction method (`native` or `ocr`) to the caller; surface conversion, rendering, and OCR errors for status/error handling.

## 4. Implement per-document processing

- [x] Implement `app/processor.py` to register each PDF with filename, path, file hash, and processing status; populate its page count as soon as parsing makes it available.
- [x] Calculate a content hash and use it to avoid reprocessing documents already marked `COMPLETE`; reuse a completed extraction when a duplicate PDF needs its own output path.
- [x] Recursively register and process every embedded PDF returned by the extractor, retaining parent/attachment context and ensuring duplicate embedded filenames cannot overwrite outputs.
- [x] Populate `page_count` after parsing and before successful completion; ensure failed or interrupted work is not recorded as complete.
- [x] Transition each document through the appropriate statuses, recording extraction method, completion time, and page count on success.
- [x] Mark a document `COMPLETE` only after its completed `.txt` output has been published successfully.
- [x] Write a plain-text `.txt` output for each successful PDF at the matching path under `data/output/` (for example, `maintenance_manual.pdf` → `maintenance_manual.txt`), including `DOCUMENT: <filename>` and the page-preserved extracted text.
- [x] Write outputs through temporary files and publish them only when complete, so interruption cannot leave a partial file that appears successful.
- [x] Mirror the input directory structure under `data/output/` and `data/failed/` so files with the same name in different input folders do not collide.
- [x] Write a neighboring `.json` metadata file with filename, file hash, page count, extraction method, and status.
- [x] On a document failure, record its error, mark it `FAILED`, copy the failed source to `data/failed/`, and leave processing able to continue with the next document.
- [x] Ensure an interrupted or failed attempt can be retried cleanly without treating partial output as a successful result.
- [x] On startup, identify documents left in `PROCESSING` by an interrupted run and return them to a retryable state without marking them complete.

## 5. Add scanning and command-line execution

- [x] Implement `app/main.py` so `python -m app.main` scans `data/input/` recursively for PDFs.
- [x] Process every PDF not already successfully processed, including newly discovered embedded PDFs, while skipping/reusing successfully completed hashes.
- [x] Print the application heading and a run summary with PDFs found, completed document count, total pages, native and OCR page totals, and failed count; print per-document filename, page count, extraction method, and clear failure reasons.
- [x] Ensure a bad or unreadable PDF does not terminate the batch; report a non-zero process exit when any PDF failed.

## 6. Validate against a representative dataset

- [x] Assemble a local synthetic test set totaling approximately 100–500 pages: native-text PDFs, scans, multi-column pages, tables, forms, poor-quality scans, engineering documents, and mixed text/image PDFs.
- [x] Include cases for embedded PDFs, duplicate file contents, nested input folders, unreadable PDFs, and interruption/restart.
- [x] Run the unit suite in an isolated ARM64 Python environment: `python -m unittest discover -s tests` (39 tests passed). The suite checks extraction routing with mocked Docling/PaddleOCR, small decorative versus significant images, page formatting, SQLite transitions/error records, output publication, cache invalidation, duplicate handling, retry behavior, and batch continuation.
- [x] Verify with live Docling and PaddleOCR that native PDFs use native extraction and scanned/no-usable-text PDFs use OCR. The offline run classified 102 pages as native and 61 as OCR.
- [x] Inspect actual runtime outputs for text usability, page separators and numbering, metadata accuracy, directory mirroring, and preservation of the original PDFs. All 11 `.txt` outputs, including the embedded PDF, had matching page headers and complete metadata; input PDFs matched their fixture hashes.
- [x] Verify database status transitions, error records, duplicate reuse, and continued batch processing after a failure. The duplicate reused completed output; the damaged PDF was the only failed record and did not stop processing.
- [x] Verify the container processes the dataset locally on the HP ZGX Nano with internet unavailable during document processing. Compose set `network_mode: none` for the extractor.
- [x] Measure representative processing speed and assess the 320,000-page workload. The 183-page synthetic corpus took about 68 seconds, including 61 CPU OCR pages. PaddleOCR used 10 CPU threads and processed OCR pages in about 0.8–1.2 seconds each; an all-OCR 320,000-page run is roughly 3.5–4.5 continuous days, before operational overhead.

## 7. Confirm POC completion and scope

- [x] Confirm text and OCR outputs are saved locally, page boundaries are preserved, processing resumes after interruption, and failures do not stop the batch. Live output inspection and unit coverage passed.
- [x] Confirm original PDFs remain untouched and no document processing calls cloud services or sends PDFs off-device. Input fixture hashes matched, and the processing container had no network.
- [x] Keep the implementation limited to PDF-to-text extraction/OCR and tracking; no out-of-scope processing or services were added.
