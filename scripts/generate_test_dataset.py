from __future__ import annotations

import argparse
import io
import shutil
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
from pypdf.errors import PdfReadError
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

PAGE_WIDTH, PAGE_HEIGHT = letter
PAGES_PER_CATEGORY = 20


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic 100-500 page PDF extraction test corpus."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/pdf_dataset"),
        help="Output directory for generated fixture PDFs",
    )
    args = parser.parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_pdf(output_dir / "01_native" / "normal_text.pdf", _native_page)
    _write_pdf(
        output_dir / "02_scanned" / "scanned.pdf",
        lambda page, pdf: _image_page(pdf, _scan_image(page, poor_quality=False)),
    )
    _write_pdf(
        output_dir / "03_columns" / "multi_column.pdf",
        _multi_column_page,
    )
    _write_pdf(output_dir / "04_tables" / "tables.pdf", _table_page)
    _write_pdf(output_dir / "05_forms" / "forms.pdf", _form_page)
    _write_pdf(
        output_dir / "06_low_quality" / "poor_quality_scans.pdf",
        lambda page, pdf: _image_page(pdf, _scan_image(page, poor_quality=True)),
    )
    _write_pdf(
        output_dir / "07_engineering" / "engineering.pdf",
        _engineering_page,
    )
    _write_pdf(
        output_dir / "08_mixed" / "mixed_text_and_images.pdf",
        _mixed_page,
    )

    duplicate_path = output_dir / "09_duplicates" / "normal_text_duplicate.pdf"
    duplicate_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        output_dir / "01_native" / "normal_text.pdf",
        duplicate_path,
    )
    _write_embedded_pdf_case(output_dir)
    unreadable_path = output_dir / "11_unreadable" / "damaged.pdf"
    unreadable_path.parent.mkdir(parents=True, exist_ok=True)
    unreadable_path.write_bytes(
        b"This file is intentionally not a valid PDF.\n"
    )

    pdfs = sorted(output_dir.rglob("*.pdf"))
    total_pages = sum(_count_pages_including_attachments(path) for path in pdfs)
    print(f"Generated {len(pdfs)} PDFs and {total_pages} valid PDF pages at {output_dir}")
    print("Coverage: native, scanned, columns, tables, forms, poor scans, engineering, mixed,")
    print("          embedded attachment, duplicate contents, nested paths, unreadable file")


def _write_pdf(
    path: Path,
    draw_page: Callable[[int, canvas.Canvas], None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=letter, pageCompression=1)
    for page_number in range(1, PAGES_PER_CATEGORY + 1):
        draw_page(page_number, pdf)
        pdf.showPage()
    pdf.save()


def _draw_lines(
    pdf: canvas.Canvas,
    lines: list[str],
    *,
    x: float = 54,
    y: float = PAGE_HEIGHT - 72,
    leading: float = 18,
) -> None:
    for line in lines:
        pdf.drawString(x, y, line)
        y -= leading


def _native_page(page_number: int, pdf: canvas.Canvas) -> None:
    _draw_lines(
        pdf,
        [
            f"LOCAL OCR TEST - NATIVE TEXT - PAGE {page_number}",
            "Document: Hydraulic System Maintenance Guide",
            "This synthetic paragraph verifies ordinary PDF text extraction.",
            f"Page {page_number} instructs technicians to inspect valve assembly V-{page_number:03}.",
            "All test wording is generated for this local validation corpus.",
        ],
    )


def _multi_column_page(page_number: int, pdf: canvas.Canvas) -> None:
    pdf.drawString(54, PAGE_HEIGHT - 54, f"MULTI-COLUMN TEST PAGE {page_number}")
    pdf.line(PAGE_WIDTH / 2, 120, PAGE_WIDTH / 2, PAGE_HEIGHT - 72)
    for row in range(1, 17):
        y = PAGE_HEIGHT - 90 - row * 28
        pdf.drawString(54, y, f"LEFT {page_number:02}-{row:02} system procedure")
        pdf.drawString(PAGE_WIDTH / 2 + 18, y, f"RIGHT {page_number:02}-{row:02} limits")


def _table_page(page_number: int, pdf: canvas.Canvas) -> None:
    pdf.drawString(54, PAGE_HEIGHT - 54, f"TABLE TEST PAGE {page_number}")
    x_positions = [54, 210, 370, 550]
    top = PAGE_HEIGHT - 90
    row_height = 28
    for row in range(8):
        y = top - row * row_height
        pdf.line(x_positions[0], y, x_positions[-1], y)
        if row < 7:
            values = (
                ("Part", "Pressure", "Inspection")
                if row == 0
                else (
                    f"P-{page_number:02}-{row:02}",
                    f"{800 + row * 25} PSI",
                    "Check seal",
                )
            )
            for x, value in zip(x_positions, values, strict=False):
                pdf.drawString(x + 4, y - 19, value)
    pdf.line(x_positions[0], top, x_positions[0], top - 7 * row_height)
    pdf.line(x_positions[-1], top, x_positions[-1], top - 7 * row_height)
    for x in x_positions[1:-1]:
        pdf.line(x, top, x, top - 7 * row_height)


def _form_page(page_number: int, pdf: canvas.Canvas) -> None:
    pdf.drawString(54, PAGE_HEIGHT - 54, f"INSPECTION FORM - PAGE {page_number}")
    fields = [
        ("Aircraft ID", f"AC-{page_number:04}"),
        ("Technician", "Morgan Lee"),
        ("Date", "2026-10-08"),
        ("Pump pressure", f"{900 + page_number} PSI"),
        ("Result", "PASS"),
    ]
    y = PAGE_HEIGHT - 120
    for label, value in fields:
        pdf.drawString(64, y + 7, label)
        pdf.rect(210, y - 2, 300, 24)
        pdf.drawString(220, y + 7, value)
        y -= 58
    pdf.rect(62, y, 16, 16)
    pdf.drawString(88, y + 3, "No leaks observed")


def _engineering_page(page_number: int, pdf: canvas.Canvas) -> None:
    _draw_lines(
        pdf,
        [
            f"ENGINEERING TEST - PAGE {page_number}",
            "Hydraulic pump assembly HPA-42",
            f"Operating pressure: {1200 + page_number} PSI",
            "Torque fitting A to 42 newton-metres.",
            "Warning: isolate power before servicing the actuator.",
        ],
    )
    pdf.setLineWidth(2)
    pdf.rect(100, 160, 110, 70)
    pdf.rect(390, 160, 110, 70)
    pdf.line(210, 195, 390, 195)
    pdf.circle(300, 195, 12)
    pdf.drawString(118, 185, "PUMP")
    pdf.drawString(405, 185, "ACTUATOR")
    pdf.drawString(280, 220, "FLOW")


def _mixed_page(page_number: int, pdf: canvas.Canvas) -> None:
    _draw_lines(
        pdf,
        [
            f"MIXED TEXT AND IMAGE TEST - PAGE {page_number}",
            "Native layer: confirm the serial number printed inside the diagram.",
        ],
    )
    image = _mixed_diagram_image(page_number)
    pdf.drawImage(
        ImageReader(image),
        70,
        240,
        width=470,
        height=280,
        preserveAspectRatio=True,
    )


def _image_page(pdf: canvas.Canvas, image: Image.Image) -> None:
    pdf.drawImage(
        ImageReader(image),
        24,
        24,
        width=PAGE_WIDTH - 48,
        height=PAGE_HEIGHT - 48,
        preserveAspectRatio=True,
    )


def _scan_image(page_number: int, *, poor_quality: bool) -> Image.Image:
    size = (420, 544) if poor_quality else (1240, 1600)
    image = Image.new("RGB", size, (232, 232, 232) if poor_quality else "white")
    draw = ImageDraw.Draw(image)
    scale = 1 if poor_quality else 3
    lines = [
        f"SCANNED MAINTENANCE PAGE {page_number}",
        "Hydraulic system inspection",
        f"Valve identifier V-{page_number:03}",
        "Verify pressure and inspect the seal.",
        "Synthetic local OCR validation document.",
    ]
    for index, line in enumerate(lines):
        draw.text((24 * scale, (45 + index * 75) * scale), line, fill=(30, 30, 30))
    if poor_quality:
        image = image.filter(ImageFilter.GaussianBlur(0.6))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=9)
        image = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    return image


def _mixed_diagram_image(page_number: int) -> Image.Image:
    image = Image.new("RGB", (1200, 720), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 100, 450, 560), outline="black", width=5)
    draw.rectangle((740, 100, 1110, 560), outline="black", width=5)
    draw.line((450, 330, 740, 330), fill="black", width=8)
    draw.text((145, 210), "SERIAL NUMBER:", fill="black")
    draw.text((150, 320), f"SN-{page_number:04}-ZX", fill="black")
    draw.text((840, 230), "PRESSURE SENSOR", fill="black")
    return image


def _write_embedded_pdf_case(output_dir: Path) -> None:
    child_bytes = io.BytesIO()
    child_pdf = canvas.Canvas(child_bytes, pagesize=letter)
    for page_number in range(1, 3):
        _draw_lines(
            child_pdf,
            [
                f"EMBEDDED REFERENCE PAGE {page_number}",
                "Attachment text must be extracted as its own document.",
            ],
        )
        child_pdf.showPage()
    child_pdf.save()
    child_bytes.seek(0)

    wrapper_path = output_dir / "10_embedded" / "manual_with_attachment.pdf"
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    writer.add_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    writer.add_attachment("reference.pdf", child_bytes.getvalue())
    with wrapper_path.open("wb") as output_file:
        writer.write(output_file)


def _count_pages_including_attachments(path: Path) -> int:
    try:
        reader = PdfReader(path)
        return len(reader.pages) + sum(
            len(PdfReader(io.BytesIO(attachment)).pages)
            for attachment_group in reader.attachments.values()
            for attachment in attachment_group
            if attachment.startswith(b"%PDF-")
        )
    except PdfReadError:
        return 0


if __name__ == "__main__":
    main()
