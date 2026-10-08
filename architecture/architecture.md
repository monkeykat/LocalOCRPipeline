Local PDF Text Extraction & OCR Pipeline
HP ZGX Nano POC
1. Goal

Build a simple, fully local pipeline on the HP ZGX Nano that:

Reads PDFs from an input directory.
Extracts existing PDF text when available.
Uses OCR for scanned/image-based PDFs.
Preserves page boundaries.
Saves the extracted text locally.
Tracks processing status and errors.

Nothing beyond text extraction is in scope.

No embeddings, chunking, vector database, RAG, LLM processing, or cloud APIs.

2. Architecture
                    /data/input
                         │
                         ▼
                  ┌─────────────┐
                  │ PDF Scanner │
                  └──────┬──────┘
                         │
                         ▼
                ┌──────────────────┐
                │ PDF has usable   │
                │ text?            │
                └────────┬─────────┘
                    YES  │  NO
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Native PDF extraction      OCR
              │                  PaddleOCR
              │                     │
              └──────────┬──────────┘
                         ▼
                  Extracted Text
                         │
                         ▼
                  /data/output
3. Technology Stack
Purpose	Technology
Language	Python
PDF processing	Docling
Native PDF extraction	Docling / underlying PDF parser
OCR	PaddleOCR
Metadata/status	SQLite
Runtime	Docker
Hardware	HP ZGX Nano, 128 GB

Everything runs locally.

No internet connection is required during document processing once the required models/packages have been installed.

4. Directory Structure
pdf-extractor/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
│
├── app/
│   ├── main.py
│   ├── processor.py
│   ├── extractor.py
│   └── database.py
│
└── data/
    ├── input/
    ├── output/
    └── failed/
Input
data/input/

Place PDFs here.

Output
data/output/

Each successfully processed PDF produces a corresponding extracted-text file.

Failed
data/failed/

PDFs that cannot be processed are recorded here or otherwise clearly identified.

5. Processing Flow

For every PDF:

Step 1: Register the document

Record:

filename
file path
file hash
page count
processing status

The file hash prevents the same PDF from being processed repeatedly.

INCLUDE EMBEDDED PDF'S WITHIN THE PDF'S

Step 2: Determine whether OCR is necessary

Attempt normal PDF text extraction.

If PDF contains usable text only:

PDF
 ↓
Native extraction
 ↓
DONE

If the PDF contains anything other than usable text:

PDF
 ↓
OCR
 ↓
DONE

The first version does not need sophisticated page-by-page classification.

A simple document-level decision is sufficient.

6. Native Text Extraction

For PDFs containing an actual text layer, extract the text directly.

Do not render these pages as images and OCR them.

This should be the fast path:

PDF
 ↓
Text layer
 ↓
Extract

The extracted text should preserve page boundaries.

Example:

=== PAGE 1 ===

This is the text from page one.

=== PAGE 2 ===

This is the text from page two.
7. OCR

For scanned PDFs or PDFs without usable text, use PaddleOCR.

The OCR pipeline should:

Render PDF pages as images.
Send pages through PaddleOCR.
Extract recognized text.
Preserve the original page number.
Combine the results into the final text file.

Example:

=== PAGE 1 ===

OCR extracted text...

=== PAGE 2 ===

OCR extracted text...

The initial implementation does not need an LLM to clean up OCR results.

8. Docling

Use Docling as the document-processing layer rather than building custom PDF parsing logic.

Docling should handle:

PDF parsing
Document structure
Page identification
Native text extraction
Coordination with OCR where appropriate

The goal is to leverage the existing document-processing capabilities rather than writing a custom PDF parser.

9. Output Format

The primary output should simply be plain text. Use the same folder structure as already exists in the Input folder.

For:

maintenance_manual.pdf

produce:

data/output/maintenance_manual.txt

Example:

DOCUMENT: maintenance_manual.pdf

=== PAGE 1 ===

Maintenance Manual
Aircraft Hydraulic System

...

=== PAGE 2 ===

Hydraulic Pump
...

=== PAGE 3 ===

Operating Limits
...

The original PDF remains untouched.

10. Metadata

Alongside the text file, optionally produce a small JSON file:

maintenance_manual.json

Example:

{
  "filename": "maintenance_manual.pdf",
  "file_hash": "abc123...",
  "page_count": 342,
  "extraction_method": "native",
  "status": "complete"
}

For OCR:

{
  "filename": "scanned_manual.pdf",
  "file_hash": "def456...",
  "page_count": 215,
  "extraction_method": "ocr",
  "status": "complete"
}

That's enough metadata for the POC.

11. SQLite Tracking

Use SQLite only to track processing.

documents
id
filename
file_hash
page_count
extraction_method
status
created_at
completed_at
errors
id
document_id
error
created_at

Statuses:

PENDING
PROCESSING
COMPLETE
FAILED

No other database is necessary.

12. Command-Line Interface

The initial application should be runnable with:

python -m app.main

It should scan:

data/input/

and process every PDF that has not already been successfully processed.

Example output:

Local PDF Extractor
===================

Found: 127 PDFs

Processing:
  manual_001.pdf     342 pages   NATIVE
  manual_002.pdf     817 pages   NATIVE
  scan_001.pdf       215 pages   OCR
  scan_002.pdf        89 pages   OCR

Completed:
  Documents: 4
  Pages:     1,463
  Native:    1,249
  OCR:         214
  Failed:       0
13. Resume Capability

The process should be safe to stop and restart.

If:

manual_001.pdf

has already successfully completed, don't process it again.

If the application crashes halfway through another document, that document should be able to restart cleanly.

This is important when eventually processing hundreds of thousands of pages.

14. Error Handling

If a PDF cannot be processed:

Record the error.
Mark the document FAILED.
Continue processing the remaining PDFs.

One bad PDF should never stop the entire batch.

Example:

ERROR: damaged_manual.pdf
Reason: Unable to read PDF

Continuing...

Completed: 126
Failed: 1
15. Initial Test Dataset

Before throwing 320,000 pages at it, test with approximately 100-500 representative pages.

Include:

Normal text PDFs
Scanned PDFs
Multi-column documents
Tables
Forms
Poor-quality scans
Engineering documents
PDFs containing mixed text and images

The goal is simply to inspect the resulting .txt files.

16. Success Criteria

The POC succeeds if:

Native PDFs produce clean, usable text.
Scanned PDFs produce reasonably accurate OCR.
Page boundaries are preserved.
The pipeline runs entirely on the HP ZGX Nano.
PDFs never leave the machine.
Processing can resume after interruption.
Failed documents don't stop the batch.
Extracted text is saved locally.
Processing speed is practical for the eventual 320K-page dataset.
Out of Scope

Do not build:

Chunking
Embeddings
Vector databases
Qdrant
RAG
LLM summarization
LLM cleanup
Web UI
Cloud APIs
Agent workflows
Distributed processing

The entire POC is simply:

PDF → Extract/OCR → Page-preserved TXT