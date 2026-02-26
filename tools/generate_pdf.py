"""Генерация PDF-отчётов из Markdown для целевых RnD-директорий."""

from __future__ import annotations

from pathlib import Path
import re
import textwrap

PAGE_WIDTH = 595
PAGE_HEIGHT = 842
LEFT = 50
TOP = 790
LINE_H = 14
MAX_CHARS = 92

TARGETS = [
    Path("01_bonferroni_aa_matching/report.md"),
    Path("02_pyspark_fast_aa/report.md"),
    Path("03_autoconfig_homogeneity_split/report.md"),
    Path("04_faiss_matcher_tradeoff/report.md"),
]


def escape_pdf_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def md_to_plain_text(md: str) -> str:
    lines: list[str] = []
    for raw in md.splitlines():
        line = raw.rstrip()
        if not line:
            lines.append("")
            continue

        if line.startswith("#"):
            title = re.sub(r"^#+\s*", "", line).strip()
            lines.append(title.upper())
            lines.append("")
            continue

        if line.lstrip().startswith(("- ", "* ")):
            item = line.lstrip()[2:].strip()
            lines.append(f"• {item}")
            continue

        if re.match(r"^\d+\)\s+", line.strip()):
            lines.append(line.strip())
            continue

        clean = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r"\1 (\2)", line)
        clean = clean.replace("**", "").replace("__", "")
        clean = clean.replace("`", "")
        lines.append(clean)

    return "\n".join(lines)


def split_pages(lines: list[str]) -> list[list[str]]:
    lines_per_page = (TOP - 60) // LINE_H
    return [lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)]


def build_content_stream(page_lines: list[str]) -> str:
    chunks = ["BT", "/F1 11 Tf", f"{LEFT} {TOP} Td"]
    for idx, line in enumerate(page_lines):
        if idx > 0:
            chunks.append(f"0 -{LINE_H} Td")
        chunks.append(f"({escape_pdf_text(line)}) Tj")
    chunks.append("ET")
    return "\n".join(chunks)


def make_pdf(text: str, output: Path) -> None:
    raw_lines: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            raw_lines.append("")
            continue
        raw_lines.extend(textwrap.wrap(paragraph, width=MAX_CHARS, break_long_words=False, break_on_hyphens=False))

    pages = split_pages(raw_lines)
    objects: list[bytes] = []

    def add_obj(data: str) -> int:
        objects.append(data.encode("latin-1", errors="replace"))
        return len(objects)

    font_obj = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_obj_ids = []

    for page_lines in pages:
        stream = build_content_stream(page_lines)
        content_obj = add_obj(f"<< /Length {len(stream.encode('latin-1', errors='replace'))} >>\nstream\n{stream}\nendstream")
        page_obj = add_obj(
            f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> /Contents {content_obj} 0 R >>"
        )
        page_obj_ids.append(page_obj)

    kids = " ".join(f"{x} 0 R" for x in page_obj_ids)
    pages_obj = add_obj(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_obj_ids)} >>")
    catalog_obj = add_obj(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>")

    for page_obj in page_obj_ids:
        objects[page_obj - 1] = objects[page_obj - 1].replace(b"/Parent 0 0 R", f"/Parent {pages_obj} 0 R".encode())

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    pdf.extend((f"trailer\n<< /Size {len(objects)+1} /Root {catalog_obj} 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n").encode("ascii"))
    output.write_bytes(pdf)


def main() -> None:
    for md_path in TARGETS:
        if not md_path.exists():
            raise FileNotFoundError(f"Не найден файл отчёта: {md_path}")
        report_text = md_to_plain_text(md_path.read_text(encoding="utf-8"))
        pdf_path = md_path.with_suffix(".pdf")
        make_pdf(report_text, pdf_path)
        print(f"Generated: {pdf_path}")


if __name__ == "__main__":
    main()
