# Local PDF Extractor

Dockerized, local PDF text extraction and OCR for the HP ZGX Nano POC. The
runtime containers have no network access. PaddleOCR and Tesseract can run
individually, and a fused mode runs Tesseract first with PaddleOCR fallback.

## Project layout

```text
app/                  Python application package
data/input/           Source PDFs
data/output_PaddleOCR/ PaddleOCR output (default)
data/output_tesseract/ Tesseract comparison output
data/output_fused/    Thresholded Tesseract/PaddleOCR output
*.cleaned.txt         Readable post-cleanup copy beside each raw transcript
data/failed/          Failed source PDFs or failure records
data/documents.sqlite3 SQLite processing and error tracking
models/               Persistent, locally cached model artifacts
```

PDFs can be organized in subfolders under `data/input/`. Each extraction
service preserves that relative folder structure under its output directory;
for example, `data/input/invoices/2025/scan.pdf` produces
`data/output_PaddleOCR/invoices/2025/scan.txt` (with its JSON and cleaned text
sidecars beside it).

## Build and prepare

Run these commands on the target machine so Docker selects its native
architecture (the HP ZGX Nano is an ARM64 system):

```sh
docker compose build extractor extractor-tesseract extractor-fused prepare-models
docker compose --profile model-setup run --rm prepare-models
```

The preparation service initializes PaddleOCR's English models under
`models/.paddleocr/`. Retain `models/` for offline PaddleOCR runs. Tesseract
and its English and orientation data are installed in the Docker image; no
model download step is needed for Tesseract. All extraction services use
Docling's native PDF parser and renderer, which do not require Docling model weights. The fused
service needs the same prepared PaddleOCR models when Tesseract's score is at
or below its fallback threshold.

## Run

Place PDFs in `data/input/`, then run the architecture's CLI:

```sh
python -m app.main
```

Or run the same CLI in the isolated Docker runtime:

```sh
docker compose run --rm extractor
```

To run the same inputs with Tesseract, keeping the PaddleOCR files intact:

```sh
docker compose run --rm extractor-tesseract
```

To run thresholded fusion:

```sh
docker compose run --rm extractor-fused
```

Fusion runs Tesseract first. If its document quality score is at or below
`OCR_FUSION_THRESHOLD` (default `90`), it runs PaddleOCR on the same OCR pages
and merges the line detections using page positions, agreement, and support
from native PDF text. Non-overlapping detections from both engines are kept.
For conflicting overlapping lines, native-text support is used when it clearly
favors one result; otherwise PaddleOCR is selected and the Tesseract alternative
is recorded for review in `fusion_assessment` in the JSON. The raw individual
engine outputs remain untouched. The fused transcript is written under
`data/output_fused/` by default. Change the threshold with, for example:

```sh
OCR_FUSION_THRESHOLD=85 docker compose run --rm extractor-fused
```

The threshold is part of the fused cache key, so changing it causes an existing
document to be reevaluated the next time the fused service runs.

The fused `quality_assessment` describes the selected output and notes that
its confidence values mix engines. `fusion_assessment` records the Tesseract
gate score, whether PaddleOCR ran, line sources, and unresolved alternatives.
Use the original PaddleOCR and Tesseract services whenever you want individual
engine outputs. Run services sequentially; they share the SQLite processing
database. To compare available outputs, including Azure reference files when
present, run:

```sh
python3 -m scripts.compare_ocr_outputs
```

If the PaddleOCR results are in a differently named folder, pass it explicitly:

```sh
python3 -m scripts.compare_ocr_outputs \
  --paddle data/output_PaddleOCR \
  --tesseract data/output_tesseract \
  --fused data/output_fused \
  --azure data/doc_intel_output
```

The report's percentages are normalized token-order similarity indicators,
not accuracy scores. It removes document/page headings and OCR position headers
and coordinates before tokenizing the text. Azure Document Intelligence is a
useful reference but not guaranteed ground truth. OCR confidence values are
recorded per engine, but their scales are not directly comparable. The report
also shows per-file OCR and full extraction times, plus aggregate speedups for
documents timed by both engines. OCR time includes engine initialization and
OCR calls; full extraction time includes PDF parsing and rendering, but excludes Docker startup
and writing output files. The first PaddleOCR document may include model-load
time in its OCR timing.

Post-cleanup is enabled by default and writes an additional `*.cleaned.txt`
beside each raw `.txt` transcript. The raw transcript remains unchanged. The
cleaned copy removes OCR position annotations, normalizes whitespace and
Unicode spacing, and drops only exact duplicate OCR lines whose page boxes
overlap almost completely. It preserves page boundaries and recognized words;
it does not guess spelling corrections. OCR lines with suspicious character
patterns or confidence below `0.80` are listed in the neighboring JSON as
`ocr_review_flags_by_page`; each flag points to its original OCR line index,
where its confidence and coordinates are recorded. Line indexes start at zero.
Set `OCR_POST_CLEANUP=false` to disable the cleaned copy and review flags; set
it back to `true` to enable them. For example:

```sh
OCR_POST_CLEANUP=false docker compose run --rm extractor
OCR_POST_CLEANUP=false docker compose run --rm extractor-tesseract
OCR_POST_CLEANUP=false docker compose run --rm extractor-fused
```

The comparison script ignores `*.cleaned.txt` sidecars so they are not counted
as separate documents.

Each output JSON also contains a `quality_assessment` with an overall 0–100
score, a score and signals for each OCR page, and a `review_recommended` flag
(default threshold: below 85). The score combines OCR confidence, agreement
between OCR lines and overlapping native PDF text when both are available, and
the rate of suspicious OCR lines. When all signals are available, the weights
are 45% confidence, 35% native-text agreement, and 20% line integrity; missing
signals are omitted and the remaining weights are rescaled. The document score
also gives the weakest OCR page extra weight. Bands are `high` (90+), `review`
(75–89.9), and `low` (below 75). It is a ranking heuristic, not an accuracy
percentage; confidence scales differ between OCR engines, so compare rankings
within an engine. The comparison report lists each engine's documents from
lowest score to highest. This only marks documents for review; it does not
send anything to Azure.

The engine and output folders can also be selected with `OCR_ENGINE` and
`OCR_OUTPUT_DIR`; the fused service uses `OCR_FUSED_OUTPUT_DIR` for its output
folder. The fallback cutoff is controlled by `OCR_FUSION_THRESHOLD`.

All runtime services run with Docker networking disabled. PaddleOCR models
must already exist in `models/`; missing artifacts cannot be downloaded at
runtime. Application processing is local; never enable remote OCR or cloud
services.

The compose service mounts `data/` and `models/` from the project directory so
outputs, failure records, SQLite status, and model caches persist across
container runs. All extraction services run as the host UID/GID, defaulting to
`1000:1000`, so generated files remain accessible to the local account. If your
account uses different IDs, set `LOCAL_UID` and `LOCAL_GID` in the environment
before running Compose commands (for example, `export LOCAL_UID="$(id -u)"`
and `export LOCAL_GID="$(id -g)"`).

The extractor uses Docling's native PDF pipeline to parse text and render
page images, and the selected OCR engine to read rendered pages. Every page
containing an embedded image is OCR'd regardless of image size or count; this
catches scans stored as many small image strips. Pages with missing or visibly
suspicious native text are OCR'd as well. Readable PDF text is retained even
when images cover most of a page; OCR lines not already represented in the
text layer are added under the correct page with their normalized page
coordinates. OCR lines, confidence, and polygons, along with native text
regions and bounding boxes, are recorded under their page numbers in the JSON
metadata. Coordinates use a top-left origin and range from 0 to 1. The output
metadata records the selected OCR engine, OCR'd page numbers, and mean
per-page confidence when available. It also records OCR and full extraction
durations in seconds. Tesseract word confidences are grouped by line and
normalized to the same 0-to-1 range for storage. pypdf discovers embedded PDF
attachments.
The attachment extraction support is pinned through `pypdf` in
`requirements.txt`.

Successful documents are written as page-preserved `.txt` files with a
neighboring `.json` metadata file under the output folder selected for the
service (`data/output_PaddleOCR/`, `data/output_tesseract/`, or
`data/output_fused/` by default).
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
classification. That behavior has since been changed: any embedded image now
causes OCR of its page, while readable native text remains available in the
output. The render scale is set near 300 dpi to preserve detail from common
300 dpi scans. Output metadata includes a pipeline version so results from
older extraction rules are regenerated automatically.

## Runtime assumptions and validation

- Python 3.11 on Debian Bookworm; the image builds for the host's native
  architecture.
- Docling Slim 2.135.0 with PDF support, PaddleOCR 2.10.0, PaddlePaddle 2.6.2,
  pytesseract 0.3.13, NumPy 1.26.4, Pyclipper 1.3.0.post6, and Rtree 1.4.1
  are pinned in `requirements.txt`. The image also installs Tesseract with
  English and orientation data. The Pyclipper pin avoids an ARM64 binary
  conflict with PaddlePaddle 2.6.2; the app loads SciPy before PaddleOCR to
  avoid its bundled zlib symbol conflict on ARM64. PaddlePaddle is the CPU
  package and publishes a Python 3.11 Linux ARM64 wheel. The built image is
  about 1.45 GB before the Tesseract packages are added.
- The validated host is ARM64 with 20 CPU cores, an NVIDIA GB10, and 121.6 GiB
  visible RAM. PaddleOCR ran on CPU; the Compose service does not request GPU
  access. Docling renders at 4 pixels per PDF point (about 288 dpi) for OCR
  input.
- Compose runs containers as the host UID/GID (default `1000:1000`) so output
  files stay accessible to the local account.
