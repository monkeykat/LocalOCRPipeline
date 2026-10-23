# Local PDF Extractor

Dockerized, local PDF text extraction and OCR for the HP ZGX Nano POC. The
runtime container has no network access. Build the image and prepare the
PaddleOCR models before processing documents offline.

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

The preparation service initializes PaddleOCR's English models under
`models/.paddleocr/`. Retain `models/` for offline runs. This pipeline uses
Docling's native PDF parser and renderer, which do not require Docling model
weights, so the image installs only the `docling-slim` PDF extra.

## Run

Place PDFs in `data/input/`, then run the architecture's CLI:

```sh
python -m app.main
```

Or run the same CLI in the isolated Docker runtime:

```sh
docker compose run --rm extractor
```

The service runs with Docker networking disabled. PaddleOCR models must
already exist in `models/`; missing artifacts cannot be downloaded at runtime.
Application processing is local; never enable remote OCR or cloud services.

The compose service mounts `data/` and `models/` from the project directory so
outputs, failure records, SQLite status, and model caches persist across
container runs. Both services run as the host UID/GID, defaulting to
`1000:1000`, so generated files remain accessible to the local account. If your
account uses different IDs, set `LOCAL_UID` and `LOCAL_GID` in the environment
before running Compose commands (for example, `export LOCAL_UID="$(id -u)"`
and `export LOCAL_GID="$(id -g)"`).

The extractor uses Docling's native PDF pipeline to parse text and render
page images without running Docling OCR. It uses PaddleOCR for documents
without usable native text and pypdf to discover embedded PDF attachments.
Small logos and seals do not force OCR; a picture covering at least 5% of a
page, or any page without usable native text, selects document-level OCR.
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

The HP ZGX Nano validation run built the ARM64 image, prepared PaddleOCR models,
and processed the synthetic corpus with container networking disabled. Ten
readable top-level PDFs plus one embedded PDF completed; the intentionally
damaged PDF was recorded as failed and did not stop the batch. Native PDFs used
native extraction, scanned and mixed PDFs used OCR, output page numbering and
metadata were checked, and input hashes were unchanged. The test batch took
about 68 seconds, including 61 OCR pages. PaddleOCR ran on CPU with its default
10 threads; its observed OCR time was about 0.8–1.2 seconds per page. A
320,000-page all-OCR run would therefore take roughly 3.5–4.5 continuous days;
native pages are faster, so the actual document mix will reduce that estimate.
Poor-quality synthetic scans had visibly noisier recognition than clean scans.
A real 28-page PDF with selectable text and a small logo exposed a false OCR
classification. The classifier now treats only substantial pictures as a
reason to OCR the whole document, and the corrected output was verified using
native text extraction. Output metadata includes a pipeline version so older
results from the previous classification rule are regenerated automatically.

## Runtime assumptions and validation

- Python 3.11 on Debian Bookworm; the image builds for the host's native
  architecture.
- Docling Slim 2.135.0 with PDF support, PaddleOCR 2.10.0, PaddlePaddle 2.6.2,
  NumPy 1.26.4, Pyclipper 1.3.0.post6, and Rtree 1.4.1 are pinned in
  `requirements.txt`. The Pyclipper pin avoids an ARM64 binary conflict with
  PaddlePaddle 2.6.2; the app also loads SciPy before PaddleOCR to avoid its
  bundled zlib symbol conflict on ARM64. PaddlePaddle is the CPU package and
  publishes a Python 3.11 Linux ARM64 wheel. The built image is about 1.45 GB.
- The validated host is ARM64 with 20 CPU cores, an NVIDIA GB10, and 121.6 GiB
  visible RAM. PaddleOCR ran on CPU; the Compose service does not request GPU
  access. Docling renders at 2x scale for OCR input.
- Compose runs containers as the host UID/GID (default `1000:1000`) so output
  files stay accessible to the local account.
