# Local PDF Extractor

Dockerized, local PDF text extraction and OCR for the HP ZGX Nano POC. The
runtime container has no network access. Install packages and prefetch model
artifacts before processing documents offline.

## Project layout

```text
app/                  Python application package
data/input/           Source PDFs
data/output/          Extracted text
data/failed/          Failed source PDFs or failure records
data/documents.sqlite3 SQLite processing and error tracking
models/               Persistent, locally cached model artifacts
```

## Build and prepare

Run these commands on the target machine so Docker selects its native
architecture (the HP ZGX Nano is an ARM64 system):

```sh
docker compose build
docker compose --profile model-setup run --rm prepare-models
```

The preparation service downloads Docling's model artifacts and initializes
the English PaddleOCR models. Artifacts are stored under `models/`, which is
mounted into the processing container and must be retained for offline runs.
The current PaddlePaddle package selection is CPU-only; verify compatibility
and performance on the target device before processing the full dataset.

## Run

Place PDFs in `data/input/`, then run the architecture's CLI:

```sh
python -m app.main
```

Or run the same CLI in the isolated Docker runtime:

```sh
docker compose run --rm extractor
```

The service runs with Docker networking disabled and Hugging Face offline mode
enabled. Model files must already exist in `models/`; missing artifacts should
cause setup/processing to fail rather than downloading them at runtime.
Application processing is local; never enable remote OCR or cloud services.

The compose service mounts `data/` and `models/` from the project directory so
outputs, failure records, SQLite status, and model caches persist across
container runs.

The extractor uses Docling's native PDF pipeline to parse text and render
page images without running Docling OCR. It uses PaddleOCR for documents
without usable native text and pypdf to discover embedded PDF attachments.
The attachment extraction support is pinned through `pypdf` in
`requirements.txt`.

Successful documents are written as page-preserved `.txt` files with a
neighboring `.json` metadata file under the matching `data/output/` path.
Failures are tracked in SQLite and the source PDF is copied under
`data/failed/`; input PDFs are left untouched. Embedded PDFs are processed
recursively and get attachment-indexed output paths beneath a directory named
for their parent PDF. Completed content hashes are reused rather than
re-extracted, and records left `PROCESSING` after an interruption are retried
when the processor starts.

The CLI recursively scans nested input folders, prints per-document progress
and a batch summary, continues after a failed PDF, and exits with a non-zero
status if one or more documents fail.

## Development validation corpus

Generate a synthetic 183-page corpus covering native text, scans, multi-column
pages, tables, forms, poor-quality scans, engineering diagrams, mixed
text/image pages, embedded PDFs, duplicates, nested directories, and an
unreadable PDF:

```sh
python -m pip install -r requirements-test.txt
python -m scripts.generate_test_dataset
```

The PDFs are written under `tests/fixtures/pdf_dataset/`; the generator can
also write to a different directory with `--output`. Run the unit suite with:

```sh
python -m unittest discover -s tests
```

The current development host can run the corpus generator and mocked
unit/integration tests, but it does not have Docling or PaddleOCR installed.
Consequently, live native extraction, OCR quality/performance, Docker offline
operation, and HP ZGX Nano behavior remain unverified until tested on the
target runtime and device.

## Runtime assumptions and validation

- Python 3.11 on Debian Bookworm; the image builds for the host's native
  architecture.
- Docling 2.135.0, PaddleOCR 2.10.0, and PaddlePaddle 2.6.2 are pinned in
  `requirements.txt`. The PaddlePaddle 2.6.2 CPU release publishes a Python
  3.11 Linux ARM64 wheel.
- Docker must be installed and working on the target system. This repository's
  development environment may not match the HP ZGX Nano; build the image and
  validate model preparation and offline processing on the target before
  treating the runtime as certified.
