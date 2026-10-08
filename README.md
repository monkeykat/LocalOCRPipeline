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

Place PDFs in `data/input/`, then run:

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
