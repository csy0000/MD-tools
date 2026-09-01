#!/usr/bin/env python
"""Render one Markdown document to PDF, with ReportLab and nothing else.

    python scripts/render_markdown_pdf.py docs/md-defaults-scientific-rationale.md \
        docs/md-defaults-scientific-rationale.pdf

Why this exists rather than pandoc: the machine this repository is developed on has no pandoc, no
LaTeX and no HTML-to-PDF engine, and installing a documentation toolchain to render one document
would add a dependency the six commands do not need. ReportLab is already present in the scientific
environment, so this is the simplest route that produces the same bytes on the same inputs.

Scope, deliberately narrow -- it renders THIS document, not arbitrary Markdown:

    ATX headings (#..####)     pipe tables with a --- separator row
    paragraphs                 fenced code blocks
    - and * bullets            > blockquotes
    1. numbered lists          --- horizontal rules
    inline **bold**, *italic*, `code`, [text](url), [@citation-key]

Anything it does not understand is emitted as body text rather than dropped, so a construct this
parser misses shows up in the PDF as visible prose instead of disappearing silently.

The Markdown file stays authoritative. This produces a rendering of it; it never edits it.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, KeepTogether, ListFlowable,
                                    ListItem, PageBreak, PageTemplate, Paragraph, Preformatted,
                                    Spacer, Table, TableStyle)
except ImportError:                                    # pragma: no cover
    raise SystemExit(
        "this renderer needs reportlab, which is not installed in the active environment.\n"
        "    conda install -c conda-forge reportlab      # or: pip install reportlab\n"
        "The Markdown source is authoritative and is unaffected; only the PDF cannot be built.")

PAGE = A4
MARGIN = 18 * mm

# ---------------------------------------------------------------------------------------------
# Fonts and glyph coverage
#
# ReportLab's built-in Helvetica is WinAnsi-encoded. A character outside that encoding is not
# rendered as a placeholder box -- it is dropped, silently. The first version of this renderer lost
# every "->" arrow out of the defaults table that way, and the PDF looked correct until it was read
# beside the Markdown. So: register a Unicode TrueType face when one is on the machine, and either
# way CHECK every character in the document against the chosen font before building.
# ---------------------------------------------------------------------------------------------

#: Searched in order. DejaVu ships with most Linux distributions and covers Latin, Greek, arrows
#: and superscripts. Only the regular and bold faces are REQUIRED: many packagings omit
#: DejaVuSans-Oblique.ttf, and silently falling back to the whole built-in family because one
#: italic file is missing is how the first version of this ended up dropping Greek letters.
_FONT_DIRECTORIES = (
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/dejavu",
    "/usr/share/fonts/TTF",
)
_FONT_FILES = {
    "regular": ("DejaVuSans.ttf",),
    "bold": ("DejaVuSans-Bold.ttf",),
    "italic": ("DejaVuSans-Oblique.ttf", "DejaVuSans-Italic.ttf"),
    "bolditalic": ("DejaVuSans-BoldOblique.ttf", "DejaVuSans-BoldItalic.ttf"),
    "mono": ("DejaVuSansMono.ttf",),
}

#: Characters that are replaced BEFORE rendering regardless of the font, because their absence is
#: silent and their ASCII form is unambiguous. Kept short and explicit: a transliteration table
#: that grows without bound is a way to quietly change what a document says.
_TRANSLITERATE = {
    "\u2192": "->",        # rightwards arrow
    "\u2248": "~",         # almost equal to
    "\u2264": "<=",
    "\u2265": ">=",
}


def _register_fonts():
    """Return the (regular, bold, italic, bold-italic, mono) family to use, and its name.

    Falls back to the built-in Helvetica/Courier family only if the regular and bold DejaVu faces
    are both absent; a missing italic is filled from the upright face rather than abandoning the
    Unicode family. Either way `check_glyph_coverage` is what stops a dropped glyph reaching the
    PDF -- this function only tries to make that check pass.
    """
    import os

    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for directory in _FONT_DIRECTORIES:
        found = {}
        for role, names in _FONT_FILES.items():
            for name in names:
                path = os.path.join(directory, name)
                if os.path.isfile(path):
                    found[role] = path
                    break
        if "regular" not in found or "bold" not in found:
            continue
        found.setdefault("italic", found["regular"])
        found.setdefault("bolditalic", found["bold"])
        found.setdefault("mono", found["regular"])
        try:
            faces = {}
            for role, path in found.items():
                face = f"DejaVu-{role}"
                pdfmetrics.registerFont(TTFont(face, path))
                faces[role] = face
            addMapping("DejaVu-regular", 0, 0, faces["regular"])
            addMapping("DejaVu-regular", 1, 0, faces["bold"])
            addMapping("DejaVu-regular", 0, 1, faces["italic"])
            addMapping("DejaVu-regular", 1, 1, faces["bolditalic"])
            return {**faces, "family": "DejaVu Sans", "source": directory}
        except Exception:
            continue
    return {"regular": "Helvetica", "bold": "Helvetica-Bold", "italic": "Helvetica-Oblique",
            "bolditalic": "Helvetica-BoldOblique", "mono": "Courier", "family": "Helvetica",
            "source": "reportlab built-in (WinAnsi)"}


def _transliterate(text: str) -> str:
    for source, target in _TRANSLITERATE.items():
        text = text.replace(source, target)
    return text


def check_glyph_coverage(text: str, fonts: dict) -> list:
    """Every character the font cannot draw, with its position. Empty is the only acceptable result.

    For a TrueType face the font's own character map is consulted. For the built-in Helvetica the
    test is WinAnsi representability, which is what ReportLab will actually encode to.
    """
    from reportlab.pdfbase import pdfmetrics

    missing = {}
    try:
        face = pdfmetrics.getFont(fonts["regular"]).face
        cmap = getattr(face, "charToGlyph", None)
    except Exception:
        cmap = None
    for index, character in enumerate(text):
        if character in "\n\r\t" or ord(character) < 128:
            continue
        if cmap is not None:
            ok = ord(character) in cmap
        else:
            try:
                character.encode("cp1252")
                ok = True
            except UnicodeEncodeError:
                ok = False
        if not ok and character not in missing:
            missing[character] = index
    return [(c, hex(ord(c)), i) for c, i in missing.items()]



# ---------------------------------------------------------------------------------------------
# Inline markup
# ---------------------------------------------------------------------------------------------

def _inline(text: str) -> str:
    """Markdown inline spans -> ReportLab's mini-HTML.

    Escaping happens FIRST and the markup is inserted afterwards, so a literal `<` in the source
    cannot become a tag and an unbalanced `*` cannot swallow the rest of the paragraph.
    """
    out = html.escape(text, quote=False)
    # Markdown backslash escapes, stashed before any markup is interpreted and restored after, so
    # `\*` survives as a literal asterisk instead of reaching the PDF as a backslash.
    escapes: list[str] = []

    def _stash_escape(match):
        escapes.append(match.group(1))
        return f"\x01{len(escapes) - 1}\x01"

    out = re.sub(r"\\([\\`*_{}\[\]()#+.!<>-])", _stash_escape, out)
    # `code` next: its contents must not be reinterpreted as bold or italic
    codes: list[str] = []

    def _stash(match):
        codes.append(match.group(1))
        return f"\x00{len(codes) - 1}\x00"

    out = re.sub(r"`([^`]+)`", _stash, out)
    out = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<link href="\2" color="#1a4b8c">\1</link>',
                 out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", out)
    for index, code in enumerate(codes):
        out = out.replace(f"\x00{index}\x00",
                          f'<font face="{_MONO[0]}" size="8.5" color="#8a2f2f">{code}</font>')
    for index, literal in enumerate(escapes):
        out = out.replace(f"\x01{index}\x01", html.escape(literal, quote=False))
    return out


#: Set once by `render`, so `_inline` can name the monospace face without threading it through
#: every call. A one-element list rather than a bare global so the assignment is visibly deliberate.
_MONO = ["Courier"]
_BOLD = ["Helvetica-Bold"]


def _plain(text: str) -> str:
    """The text as it will be SEEN: markup removed, so a width measurement measures the glyphs."""
    out = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"\1", text)
    out = re.sub(r"\\([\\`*_{}\[\]()#+.!<>-])", r"\1", out)
    out = out.replace("**", "").replace("`", "")
    return re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"\1", out)


CITATION = re.compile(r"\[@([^\]]+)\]")


def _citations(text: str) -> str:
    """`[@key1; @key2]` -> `[key1; key2]`, so a key is readable without a bibliography engine."""
    def render(match):
        keys = [k.strip().lstrip("@") for k in match.group(1).split(";")]
        return "[" + "; ".join(keys) + "]"
    return CITATION.sub(render, text)


# ---------------------------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------------------------

def _styles(fonts):
    base = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=base["BodyText"], fontName=fonts["regular"],
                          fontSize=9.4,
                          leading=13.2, spaceBefore=0, spaceAfter=6, alignment=TA_LEFT)
    return {
        "body": body,
        "h1": ParagraphStyle("H1", parent=body, fontName=fonts["bold"], fontSize=17,
                             leading=21, spaceBefore=2, spaceAfter=10,
                             textColor=colors.HexColor("#14243d")),
        "h2": ParagraphStyle("H2", parent=body, fontName=fonts["bold"], fontSize=13,
                             leading=17, spaceBefore=15, spaceAfter=7,
                             textColor=colors.HexColor("#14243d")),
        "h3": ParagraphStyle("H3", parent=body, fontName=fonts["bold"], fontSize=10.8,
                             leading=14.5, spaceBefore=11, spaceAfter=5,
                             textColor=colors.HexColor("#26405f")),
        "h4": ParagraphStyle("H4", parent=body, fontName=fonts["bolditalic"], fontSize=9.8,
                             leading=13, spaceBefore=9, spaceAfter=4),
        "quote": ParagraphStyle("Quote", parent=body, leftIndent=10, rightIndent=6,
                                fontName=fonts["italic"],
                                textColor=colors.HexColor("#3a3a3a"),
                                borderPadding=(0, 0, 0, 6), spaceBefore=4, spaceAfter=8),
        "item": ParagraphStyle("Item", parent=body, spaceAfter=3),
        "cell": ParagraphStyle("Cell", parent=body, fontSize=7.5, leading=9.8, spaceAfter=0),
        "cellhead": ParagraphStyle("CellHead", parent=body, fontSize=7.5, leading=9.8,
                                   spaceAfter=0, fontName=fonts["bold"],
                                   textColor=colors.white),
        "code": ParagraphStyle("Code", parent=body, fontName=fonts["mono"], fontSize=7.6,
                               leading=9.6),
    }


# ---------------------------------------------------------------------------------------------
# Block parsing
# ---------------------------------------------------------------------------------------------

def _table(rows: list[str], style, width: float):
    """A pipe table. The separator row is dropped; every cell is a wrapped Paragraph.

    Column widths are proportional to the longest cell in each column, floored so a narrow column
    never collapses to an unreadable ribbon, and normalised to the frame width so a wide table
    cannot run off the page -- which is the overflow this renderer exists to avoid.
    """
    grid = []
    for line in rows:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(set(c) <= set("-: ") and c for c in cells):
            continue                                    # the --- separator row
        grid.append(cells)
    if not grid:
        return None
    columns = max(len(r) for r in grid)
    grid = [r + [""] * (columns - len(r)) for r in grid]

    # Two constraints, and the second is the one that produced "electrosta / tics" before it
    # existed: a column must be wide enough for its longest UNBREAKABLE token, or ReportLab breaks
    # mid-word. So a per-column floor is measured in points from the longest word actually present,
    # and only the space left over is shared out in proportion to total content.
    from reportlab.pdfbase.pdfmetrics import stringWidth

    padding = 9.0                                       # LEFTPADDING + RIGHTPADDING, plus a hair
    floors, weights = [], []
    for index in range(columns):
        longest_word = 0.0
        content = 0
        for row in grid:
            plain = _plain(_citations(row[index]))
            content = max(content, len(plain))
            for word in plain.split():
                longest_word = max(longest_word, stringWidth(word, _BOLD[0], 7.5))
        # Capped: one very long path must not be allowed to starve four other columns. Above the
        # cap the token breaks mid-word, which is visible and therefore honest, rather than pushing
        # the table off the page.
        floors.append(min(longest_word + padding, width * 0.30))
        weights.append(max(content, 6))

    spare = width - sum(floors)
    if spare <= 0:
        # Every column is at its floor and they still do not fit. Scale proportionally and accept
        # the breaks rather than running off the page, which is the worse failure.
        scale = width / sum(floors)
        widths = [f * scale for f in floors]
    else:
        total = sum(weights)
        widths = [f + spare * w / total for f, w in zip(floors, weights)]

    # A `| | |` header -- the shape a metadata block takes -- would otherwise render as a bare dark
    # stripe with nothing in it. An empty header row is dropped instead.
    headed = any(c.strip() for c in grid[0])
    if not headed:
        grid = grid[1:]
        if not grid:
            return None

    data = ([[Paragraph(_inline(_citations(c)), style["cellhead"]) for c in grid[0]]]
            if headed else [])
    data += [[Paragraph(_inline(_citations(c)), style["cell"]) for c in row]
             for row in (grid[1:] if headed else grid)]

    table = Table(data, colWidths=widths, repeatRows=1 if headed else 0, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#26405f") if headed else colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b9c4d1")),
        ("ROWBACKGROUNDS", (0, 1 if headed else 0), (-1, -1),
         [colors.white, colors.HexColor("#f2f5f8")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def build_story(markdown: str, style, width: float) -> list:
    story: list = []
    lines = markdown.replace("\r\n", "\n").split("\n")
    index, total = 0, len(lines)

    def flush(buffer: list[str]):
        if not buffer:
            return
        text = " ".join(b.strip() for b in buffer).strip()
        if text:
            story.append(Paragraph(_inline(_citations(text)), style["body"]))
        buffer.clear()

    paragraph: list[str] = []
    while index < total:
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            flush(paragraph)
            index += 1
            continue

        if stripped.startswith("```"):
            flush(paragraph)
            index += 1
            block = []
            while index < total and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1
            story.append(Spacer(1, 3))
            story.append(Preformatted("\n".join(block), style["code"]))
            story.append(Spacer(1, 6))
            continue

        if re.fullmatch(r"-{3,}", stripped):
            flush(paragraph)
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.6,
                                    color=colors.HexColor("#b9c4d1")))
            story.append(Spacer(1, 6))
            index += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            flush(paragraph)
            level = len(heading.group(1))
            story.append(Paragraph(_inline(_citations(heading.group(2))), style[f"h{level}"]))
            index += 1
            continue

        if stripped.startswith("|"):
            flush(paragraph)
            rows = []
            while index < total and lines[index].strip().startswith("|"):
                rows.append(lines[index])
                index += 1
            table = _table(rows, style, width)
            if table is not None:
                story.append(Spacer(1, 2))
                story.append(table)
                story.append(Spacer(1, 8))
            continue

        if stripped.startswith(">"):
            flush(paragraph)
            block = []
            while index < total and lines[index].strip().startswith(">"):
                block.append(lines[index].strip().lstrip(">").strip())
                index += 1
            story.append(Paragraph(_inline(_citations(" ".join(block))), style["quote"]))
            continue

        bullet = re.match(r"^(\s*)([-*])\s+(.*)$", line)
        number = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if bullet or number:
            flush(paragraph)
            ordered = number is not None
            items, current = [], None
            while index < total:
                nxt = lines[index]
                match = (re.match(r"^(\s*)(\d+)\.\s+(.*)$", nxt) if ordered
                         else re.match(r"^(\s*)([-*])\s+(.*)$", nxt))
                if match:
                    if current is not None:
                        items.append(" ".join(current))
                    current = [match.group(3)]
                    index += 1
                elif nxt.strip() and nxt.startswith((" ", "\t")) and current is not None:
                    current.append(nxt.strip())         # a continuation line of the same item
                    index += 1
                else:
                    break
            if current is not None:
                items.append(" ".join(current))
            story.append(ListFlowable(
                [ListItem(Paragraph(_inline(_citations(t)), style["item"]), leftIndent=16)
                 for t in items],
                bulletType="1" if ordered else "bullet",
                bulletFormat="%s." if ordered else None,
                bulletFontSize=8, leftIndent=18, spaceBefore=1, spaceAfter=6))
            continue

        paragraph.append(line)
        index += 1

    flush(paragraph)
    return story


# ---------------------------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------------------------

def render(source: Path, target: Path) -> dict:
    markdown = _transliterate(Path(source).read_text(encoding="utf-8"))
    fonts = _register_fonts()
    _MONO[0] = fonts["mono"]
    _BOLD[0] = fonts["bold"]

    missing = check_glyph_coverage(markdown, fonts)
    if missing:
        raise SystemExit(
            f"{fonts['family']} ({fonts['source']}) cannot draw "
            f"{len(missing)} character(s) in {source}. ReportLab DROPS an unrepresentable "
            f"character silently, so the PDF would be missing text without looking broken:\n  "
            + "\n  ".join(f"{c!r} ({code}) at offset {i}" for c, code, i in missing)
            + "\n Install a Unicode font (DejaVu) or add an entry to _TRANSLITERATE.")

    style = _styles(fonts)
    width = PAGE[0] - 2 * MARGIN

    document = BaseDocTemplate(str(target), pagesize=PAGE,
                               leftMargin=MARGIN, rightMargin=MARGIN,
                               topMargin=MARGIN, bottomMargin=MARGIN + 6 * mm,
                               title=Path(source).stem, author="MD-tools")
    frame = Frame(MARGIN, MARGIN + 6 * mm, width, PAGE[1] - 2 * MARGIN - 6 * mm, id="body",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    footer_text = f"{Path(source).name}  ·  MD-tools"

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#6b7785"))
        canvas.drawString(MARGIN, MARGIN, footer_text)
        canvas.drawRightString(PAGE[0] - MARGIN, MARGIN, f"page {doc.page}")
        canvas.restoreState()

    document.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=footer)])
    story = build_story(markdown, style, width)
    document.build(story)
    return {"source": str(source), "target": str(target), "pages": document.page,
            "flowables": len(story), "font": fonts["family"], "font_source": fonts["source"]}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(__doc__.strip().splitlines()[0]
                         + "\n\n    python scripts/render_markdown_pdf.py <input.md> <output.pdf>")
    result = render(Path(argv[0]), Path(argv[1]))
    print(f"  wrote {result['target']}: {result['pages']} pages "
          f"from {result['flowables']} blocks, font {result['font']} ({result['font_source']}), "
          f"every glyph covered")
    return 0


if __name__ == "__main__":                             # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
