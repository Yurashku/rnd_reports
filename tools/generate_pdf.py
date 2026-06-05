"""Генерация PDF-отчётов из Markdown для целевых RnD-директорий.

Рендер через matplotlib (шрифт DejaVu Sans) — корректно отображает **кириллицу**,
рисует markdown-таблицы настоящими сетками и встраивает картинки ``![alt](path)``.
Цели находятся автоматически: каждый ``rnd/<topic>/report.md`` подхватывается без
правки списка. Интерфейс прежний::

    python tools/generate_pdf.py
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

RND_ROOT = Path("rnd")

# Геометрия A4 в пунктах (1 in = 72 pt). Координаты — доля фигуры (0..1).
PAGE_W_PT, PAGE_H_PT = 595.0, 842.0
FIG_W_IN, FIG_H_IN = PAGE_W_PT / 72.0, PAGE_H_PT / 72.0
LEFT, RIGHT, TOP, BOTTOM = 0.07, 0.95, 0.95, 0.06
USABLE_W = RIGHT - LEFT

# Шрифты (DejaVu идёт в комплекте matplotlib — полная кириллица).
SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

# Размеры шрифтов по типам блоков.
SIZE_BODY = 10.0
SIZE_H = {1: 17.0, 2: 13.5, 3: 11.5, 4: 10.5}
SIZE_CODE = 8.5
SIZE_TABLE = 8.0


def discover_targets() -> list[Path]:
    return sorted(RND_ROOT.glob("*/report.md"))


def _pt2fracy(pt: float) -> float:
    return pt / PAGE_H_PT


# Эмодзи/пиктограммы вне покрытия DejaVu — убираем, чтобы не было «tofu»-боксов.
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U00002B00-\U00002BFF\U000023E0-\U000023FF\U0000FE0F]"
)


def _strip_inline(s: str) -> str:
    """Убрать markdown-разметку для plain-рендера (жирный/код/ссылки/эмодзи)."""
    s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"", s)          # картинки убираем из текста
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)          # [text](url) -> text
    s = s.replace("**", "").replace("__", "").replace("`", "")
    s = _EMOJI.sub("", s)
    return s.strip()


def _wrap(s: str, size: float, width_frac: float) -> list[str]:
    usable_pt = width_frac * PAGE_W_PT
    nchars = max(8, int(usable_pt / (0.54 * size)))
    out: list[str] = []
    for part in s.split("\n"):
        out.extend(textwrap.wrap(part, nchars) or [""])
    return out or [""]


# --------------------------------------------------------------------------- #
# Парсер markdown → блоки
# --------------------------------------------------------------------------- #

_IMG = re.compile(r"^\s*!\[([^\]]*)\]\(([^)]+)\)\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def _split_cells(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [c.strip() for c in row.split("|")]


def parse_blocks(md: str) -> list[dict]:
    lines = md.splitlines()
    blocks: list[dict] = []
    para: list[str] = []

    def flush_para():
        if para:
            blocks.append({"type": "para", "text": " ".join(para)})
            para.clear()

    i, n = 0, len(lines)
    while i < n:
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        # fenced code
        if stripped.startswith("```"):
            flush_para()
            code: list[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            blocks.append({"type": "code", "lines": code})
            continue

        # image
        m = _IMG.match(line)
        if m:
            flush_para()
            blocks.append({"type": "image", "alt": m.group(1), "path": m.group(2).strip()})
            i += 1
            continue

        # table: header row + separator
        if "|" in line and i + 1 < n and _TABLE_SEP.match(lines[i + 1]) and "|" in lines[i + 1]:
            flush_para()
            headers = _split_cells(line)
            i += 2
            rows: list[list[str]] = []
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append([_strip_inline(c) for c in _split_cells(lines[i])])
                i += 1
            blocks.append({"type": "table",
                           "headers": [_strip_inline(h) for h in headers], "rows": rows})
            continue

        if not stripped:
            flush_para()
            i += 1
            continue

        if stripped.startswith("#"):
            flush_para()
            level = len(stripped) - len(stripped.lstrip("#"))
            blocks.append({"type": "heading", "level": min(level, 4),
                           "text": _strip_inline(stripped.lstrip("#").strip())})
            i += 1
            continue

        if re.match(r"^[-*]\s+", stripped):
            flush_para()
            blocks.append({"type": "bullet", "text": _strip_inline(stripped[2:].strip())})
            i += 1
            continue

        if re.match(r"^\d+[.)]\s+", stripped):
            flush_para()
            blocks.append({"type": "num", "text": _strip_inline(stripped)})
            i += 1
            continue

        if stripped.startswith(">"):
            flush_para()
            blocks.append({"type": "quote", "text": _strip_inline(stripped.lstrip(">").strip())})
            i += 1
            continue

        if set(stripped) <= {"-", "*", "_"} and len(stripped) >= 3:
            flush_para()
            blocks.append({"type": "hr"})
            i += 1
            continue

        para.append(_strip_inline(stripped))
        i += 1

    flush_para()
    return blocks


# --------------------------------------------------------------------------- #
# Рендер блоков на страницы
# --------------------------------------------------------------------------- #

class _Renderer:
    def __init__(self, pdf: PdfPages):
        self.pdf = pdf
        self.fig = None
        self.ax = None
        self.y = 0.0
        self._new_page()

    def _new_page(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")
        self.y = TOP

    def _ensure(self, need_frac: float):
        if self.y - need_frac < BOTTOM:
            self._new_page()

    def _text(self, x, s, size, *, weight="normal", family=SANS, color="black"):
        self.ax.text(x, self.y, s, fontsize=size, fontweight=weight, family=family,
                     color=color, va="top", ha="left", transform=self.ax.transAxes)

    def _advance(self, pt):
        self.y -= _pt2fracy(pt)

    def gap(self, pt=4.0):
        self._advance(pt)

    def lines(self, text, size, x=LEFT, *, weight="normal", family=SANS,
              color="black", width=USABLE_W, lead=1.45):
        for ln in _wrap(text, size, width):
            self._ensure(_pt2fracy(size * lead))
            self._text(x, ln, size, weight=weight, family=family, color=color)
            self._advance(size * lead)

    # --- блоки -------------------------------------------------------------
    def heading(self, level, text):
        self.gap(8 if level <= 2 else 5)
        size = SIZE_H.get(level, SIZE_BODY)
        self.lines(text, size, weight="bold", lead=1.5)
        if level <= 2:  # подчёркивание под крупными заголовками
            self._ensure(_pt2fracy(4))
            self.ax.plot([LEFT, RIGHT], [self.y, self.y], color="0.55", lw=0.8,
                         transform=self.ax.transAxes)
            self._advance(4)

    def paragraph(self, text):
        self.lines(text, SIZE_BODY)
        self.gap(3)

    def bullet(self, text):
        self._ensure(_pt2fracy(SIZE_BODY * 1.45))
        self._text(LEFT, "•", SIZE_BODY, weight="bold")
        # текст с отступом, перенос по доступной ширине
        x = LEFT + 0.02
        for k, ln in enumerate(_wrap(text, SIZE_BODY, USABLE_W - 0.02)):
            self._ensure(_pt2fracy(SIZE_BODY * 1.45))
            self._text(x, ln, SIZE_BODY)
            self._advance(SIZE_BODY * 1.45)

    def numbered(self, text):
        self.lines(text, SIZE_BODY, x=LEFT + 0.01)

    def quote(self, text):
        self.lines(text, SIZE_BODY, x=LEFT + 0.02, color="0.25", width=USABLE_W - 0.04)
        self.gap(3)

    def code(self, code_lines):
        self.gap(3)
        for ln in code_lines:
            for sub in _wrap(ln, SIZE_CODE, USABLE_W) if ln else [""]:
                self._ensure(_pt2fracy(SIZE_CODE * 1.4))
                self._text(LEFT + 0.01, sub, SIZE_CODE, family=MONO, color="0.15")
                self._advance(SIZE_CODE * 1.4)
        self.gap(3)

    def hr(self):
        self.gap(3)
        self._ensure(_pt2fracy(4))
        self.ax.plot([LEFT, RIGHT], [self.y, self.y], color="0.8", lw=0.6,
                     transform=self.ax.transAxes)
        self._advance(6)

    def image(self, path, base_dir):
        p = (base_dir / path).resolve()
        if not p.exists():
            self.lines(f"[нет картинки: {path}]", SIZE_BODY, color="0.5")
            return
        try:
            img = plt.imread(str(p))
        except Exception:  # noqa: BLE001
            self.lines(f"[не прочитать: {path}]", SIZE_BODY, color="0.5")
            return
        ih, iw = img.shape[0], img.shape[1]
        w_frac = min(USABLE_W, 0.74)
        h_frac = w_frac * (ih / iw) * (PAGE_W_PT / PAGE_H_PT)
        if h_frac > (TOP - BOTTOM) * 0.9:  # очень высокая картинка — ужать
            h_frac = (TOP - BOTTOM) * 0.9
            w_frac = h_frac * (iw / ih) * (PAGE_H_PT / PAGE_W_PT)
        self.gap(4)
        self._ensure(h_frac + _pt2fracy(6))
        x = LEFT + (USABLE_W - w_frac) / 2
        iax = self.fig.add_axes([x, self.y - h_frac, w_frac, h_frac])
        iax.imshow(img)
        iax.axis("off")
        self._advance(h_frac * PAGE_H_PT + 6)

    def table(self, headers, rows):
        ncol = len(headers)
        if ncol == 0:
            return
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        # Моноширинный шрифт: ширина символа предсказуема → колонки без наложений.
        # Колонки могут переноситься максимум на 2 строки (cap по символам).
        cap = 22
        colw = [min(cap, max([len(headers[j])] + [len(r[j]) for r in rows]) or 1)
                for j in range(ncol)]
        total = sum(colw) + ncol
        char_frac = USABLE_W / total
        # Подбираем размер так, чтобы W символов уместились в (W+1) ячеек (моно ≈0.62*size).
        size = min(SIZE_TABLE, char_frac * PAGE_W_PT / 0.62)
        size = max(5.5, size)
        xs = [LEFT]
        for w in colw:
            xs.append(xs[-1] + (w + 1) * char_frac)

        def wrap_cell(c, w):
            return textwrap.wrap(c, max(4, w)) or [""]

        def row_h(cells):
            return max((len(wrap_cell(c, colw[j])) for j, c in enumerate(cells)),
                       default=1) * size * 1.45

        self.gap(4)
        self._hline()
        self._table_row(headers, xs, colw, size, bold=True)
        self._hline()
        for r in rows:
            if self.y - _pt2fracy(row_h(r) + 2) < BOTTOM:
                self._new_page()
                self._hline()
                self._table_row(headers, xs, colw, size, bold=True)
                self._hline()
            self._table_row(r, xs, colw, size, bold=False)
        self._hline()
        self.gap(5)

    def _hline(self):
        self._ensure(_pt2fracy(3))
        self.ax.plot([LEFT, RIGHT], [self.y, self.y], color="0.6", lw=0.7,
                     transform=self.ax.transAxes)
        self._advance(2)

    def _table_row(self, cells, xs, colw, size, *, bold):
        wrapped = [textwrap.wrap(c, max(4, colw[j])) or [""] for j, c in enumerate(cells)]
        nlines = max((len(w) for w in wrapped), default=1)
        self._ensure(_pt2fracy(nlines * size * 1.45))
        y0 = self.y
        for li in range(nlines):
            for j, w in enumerate(wrapped):
                if li < len(w):
                    self._text(xs[j] + 0.004, w[li], size, family=MONO,
                               weight="bold" if bold else "normal")
            self._advance(size * 1.45)
        for x in xs:  # вертикальные разделители колонок
            self.ax.plot([x, x], [y0, self.y], color="0.8", lw=0.4,
                         transform=self.ax.transAxes)

    def finish(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
            self.fig = None


# Детерминированные метаданные: без них matplotlib пишет текущий CreationDate, и каждый
# прогон даёт новый PDF-байт-поток (лишние git-диффы по всем report.pdf).
_PDF_METADATA = {"Creator": "rnd_reports/tools/generate_pdf.py", "Producer": "matplotlib",
                 "CreationDate": None}


def render_pdf(md_path: Path, pdf_path: Path) -> None:
    blocks = parse_blocks(md_path.read_text(encoding="utf-8"))
    base_dir = md_path.parent
    with PdfPages(pdf_path, metadata=_PDF_METADATA) as pdf:
        r = _Renderer(pdf)
        for b in blocks:
            t = b["type"]
            if t == "heading":
                r.heading(b["level"], b["text"])
            elif t == "para":
                r.paragraph(b["text"])
            elif t == "bullet":
                r.bullet(b["text"])
            elif t == "num":
                r.numbered(b["text"])
            elif t == "quote":
                r.quote(b["text"])
            elif t == "code":
                r.code(b["lines"])
            elif t == "table":
                r.table(b["headers"], b["rows"])
            elif t == "image":
                r.image(b["path"], base_dir)
            elif t == "hr":
                r.hr()
        r.finish()


def main() -> None:
    targets = discover_targets()
    if not targets:
        raise FileNotFoundError(f"Не найдены report.md в {RND_ROOT}/*/")
    for md_path in targets:
        pdf_path = md_path.with_suffix(".pdf")
        render_pdf(md_path, pdf_path)
        print(f"Generated: {pdf_path}")


if __name__ == "__main__":
    main()
