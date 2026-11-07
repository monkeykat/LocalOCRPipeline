# Local PDF Extractor

Dockerized, local PDF text extraction and OCR for the HP ZGX Nano POC. The
runtime containers have no network access. PaddleOCR and Tesseract can run
individually, and a fused mode runs Tesseract first with PaddleOCR fallback.

## Project layout

```text
app/                  Python application package
documents/            Source PDFs from AviationRegs
data/output_PaddleOCR/ PaddleOCR output (default)
data/output_tesseract/ Tesseract comparison output
data/output_fused/    Thresholded Tesseract/PaddleOCR output
*.cleaned.txt         Readable post-cleanup copy beside each raw transcript
data/failed/          Failed source PDFs or failure records
data/documents.sqlite3 SQLite processing and error tracking
models/               Persistent, locally cached model artifacts
```

The default input is
`/home/gulfaero/Documents/Code/AviationRegs/AviationRegs/documents/`. Docker
mounts it read-only at `/app/input`. Override the host source with
`OCR_SOURCE_DIR=/path/to/documents`; direct Python runs can set `OCR_INPUT_DIR`.

PDFs can be organized in subfolders under the source directory. Each extraction
service preserves that relative folder structure under its output directory;
for example, `documents/FAA/AC/scan.pdf` produces
`data/output_PaddleOCR/FAA/AC/scan.txt` (with its JSON and cleaned text
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

Run the architecture's CLI against the default AviationRegs documents folder:

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

Each batch uses up to two worker processes by default. Set `OCR_WORKERS` to
change the number of PDFs processed at once; for example:

```sh
OCR_WORKERS=4 docker compose run --rm extractor-fused
```

Each worker runs its own extractor and OCR models, so memory use can rise with
the worker count. Run one batch with the desired worker count instead of
starting multiple extractor containers against the same input and output
folders.

If a worker process crashes, the batch retries its in-flight PDFs once and
restarts that pool with the configured worker count. A document that fails
again is recorded as failed and the rest of the pool continues. After five
pool crashes in a row without a successful document, only that pool pauses;
its remaining PDFs stay pending for the next run, while another pool can
finish. If the operating system kills the whole container, rerun the same
command; the database resets interrupted documents so processing resumes.

To profile native and OCR workloads separately, run a single worker:

```sh
OCR_WORKERS=1 docker compose run --rm extractor-fused
```

Single-worker run metrics include per-document-type CPU and memory samples.
Fused documents are classified as OCR if any page used OCR. Cached documents
are counted separately and excluded from the resource profile. The profile
includes the shared process baseline, so use its maximum memory figures as a
conservative one-worker footprint when choosing concurrency. Separate-pool
runs also report independent pool CPU and worker-process-tree RSS measurements;
the overall cgroup measurements still show total container use.

After profiling, separate pool limits can be set independently:

```sh
OCR_NATIVE_WORKERS=1 OCR_OCR_WORKERS=1 docker compose run --rm extractor-fused
```

Set both variables to choose exact caps. For example,
`OCR_NATIVE_WORKERS=1 OCR_OCR_WORKERS=2` allows at most one native-pool job and
two OCR-pool jobs at once (three workers total). If only one variable is set,
the other pool uses `OCR_WORKERS` (default `2`). If neither is set,
`OCR_WORKERS` retains the shared-pool behavior.

Setting both limits to `3` allows up to six PDFs in flight: three assigned to
the native pool and three assigned to the OCR pool.

Separate-pool mode preflights PDFs before processing: only high-confidence
text-only PDFs go to the native pool; image-heavy, uncertain, encrypted, or
unreadable PDFs go to the OCR pool. The startup output and run metrics report
the predicted native/OCR counts and each pool's worker cap. The extractor
checks each result afterward and logs any native-to-OCR misclassification,
which can make actual OCR use exceed the OCR-pool cap for that document. The
preflight classification is committed to SQLite as each PDF is classified,
keyed by content hash and classifier version. Later runs reuse that
classification for unchanged PDFs. If extraction later shows that a PDF routed
to native needed OCR, the database updates its saved classification to OCR for
the next run. The startup output and run metrics show
how many classifications were reused versus computed. Increment
`CLASSIFIER_VERSION` when changing classifier rules so old cached decisions are
recomputed. The preflight scans the input list once before worker pools start.

In separate-pool mode, console progress reports an OCR ETA after each OCR-pool
document finishes. It uses the average elapsed time per completed OCR-pool
document and the configured OCR worker count to estimate the remaining queue;
the estimate improves as more documents finish. Run metrics include the OCR
document total, completed count, average duration, estimated remaining seconds,
and estimated completion time. These estimates cover PDFs routed to the OCR
pool, including uncertain classifications.

Separate-pool run metrics sample CPU and worker-process-tree RSS independently
for the native and OCR pools, including the largest single worker tree. The
container-wide cgroup measurements remain the authoritative total. Process RSS
can count shared pages more than once and does not include container page
cache; GPU measurements remain device-wide.

For a rough memory estimate from a separate-pool run, multiply each pool's
`max_single_worker_process_tree_rss_bytes` by its worker cap, then compare the
sum with the container-wide `memory.max_bytes` and leave headroom. This can
overestimate shared pages and does not predict unusually large PDFs. Pool CPU
figures show the sampled total cores used by each pool at the configured caps;
leave headroom for short peaks between samples. Review native-to-OCR mismatch
warnings before raising the native cap.

Each run creates a resource summary JSON file under `data/run_metrics/` at
startup, overwrites it every 10 minutes, and writes a final summary at
shutdown. CPU and memory are sampled every second; NVIDIA GPU telemetry is
sampled every five seconds when available. The file records the running
averages and maxima, worker count, document progress, and run duration; it
stores only the summary, not individual samples. CPU percentages are relative
to the CPU capacity available to the container. GPU readings are device-wide
and can include activity from other host processes.

Fusion runs Tesseract first. It runs PaddleOCR on any OCR page whose Tesseract
quality score is at or below `OCR_FUSION_THRESHOLD` (default `90`), then
compares their page scores. When one engine scores at least 85 and leads by at
least 8 points, the fused output uses that engine's full page result. Other
pages merge detections using page positions, agreement, and support from native
PDF text. Non-overlapping detections are retained when neither engine clearly
leads. For conflicting lines, native-text support is used when it clearly
favors one result; otherwise PaddleOCR is selected and the Tesseract alternative
is recorded for review in `fusion_assessment` in the JSON. The raw individual
engine outputs remain untouched. The fused transcript is written under
`data/output_fused/` by default. Change the fallback threshold with, for example:

```sh
OCR_FUSION_THRESHOLD=85 docker compose run --rm extractor-fused
```

The threshold is part of the fused cache key, so changing it causes an existing
document to be reevaluated the next time the fused service runs.
The page-selection algorithm version is also checked for documents that used
Paddle fallback, so those cached OCR outputs are reevaluated after a fusion
selection change. Native-only fused outputs remain cached.

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
re-extracted. The processor checks the expected output sidecar directly before
searching other outputs for duplicate content, keeping normal cache hits fast.
At run start, every discovered PDF is hashed and registered in SQLite; files
with a valid `COMPLETE` record and current output are excluded from preflight
classification and worker queues. Changed or stale files are queued again.
Records left `PROCESSING` after an interruption are retried when the processor
starts.

## Native text quality review

Scan existing native output sidecars without starting OCR:

```sh
python -m app.audit_native_text data/output_fused
```

The audit reports pages with empty or suspicious native text, including long
text layers with almost no alphabetic characters. The same page-level check is
used during extraction; flagged pages are sent to OCR, and OCR replaces a
corrupt native text layer for that page. Each JSON sidecar records the check's
page counts, alphabetic ratios, and reasons under `native_text_assessment`.
This is a triage heuristic, so review flagged pages against their PDFs before
treating the OCR result as authoritative.

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
  visible RAM. PaddleOCR uses the CPU build. The fused Compose service exposes
  the GPU to `nvidia-smi` for telemetry only; extraction still runs on CPU.
  Docling renders at 4 pixels per PDF point (about 288 dpi) for OCR input.
- Compose runs containers as the host UID/GID (default `1000:1000`) so output
  files stay accessible to the local account.
