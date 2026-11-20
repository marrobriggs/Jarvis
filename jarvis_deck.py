"""
jarvis_deck: client slide decks in the Jarvis look.

Jarvis copies this file (and jarvis_theme.py) into each job's work folder.
The analyst fills in content; this module handles every layout, color and
font decision, so decks look consistent and the analyst's code stays short.

Charts are native PowerPoint charts (editable by the client), colored with
the same validated palette as the Streamlit reports.

Usage:
    import jarvis_deck as jd
    deck = jd.Deck("Maven Toys: 2017-2018 Review", "Sales, profit and inventory across 50 stores")
    deck.kpis("The headline numbers", [("Revenue", "$14.4M"), ("Profit", "$4.0M"),
                                       ("Margin", "27.8%"), ("Stores", "50")])
    deck.bar("Toys earn the most profit", ["Toys", "Art & Crafts", "Games"], [1.08e6, 0.75e6, 0.67e6],
             takeaway="Toys bring in 27% of all profit.", number_format="$#,##0",
             value_title="Profit ($)", highlight="Toys")
    deck.grouped_bar("Revenue vs profit by category", cats, {"Revenue": rev, "Profit": prof}, ...)
    deck.line("Monthly profit", months, {"Profit": values}, takeaway="...", number_format="$#,##0")
    deck.table("Stores to restock first", df.head(10), takeaway="...")
    deck.bullets("Recommended actions", ["...", "..."])
    deck.caveats(["Profit excludes rent and staff costs.", "..."])
    deck.save("outputs/report.pptx")
"""

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

import jarvis_theme as jt

FONT = "Segoe UI"
W, H = Inches(13.333), Inches(7.5)          # 16:9
MARGIN = Inches(0.6)


def _rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color.lstrip("#"))


def _lines(text, size_pt, width_in):
    """Rough count of how many lines text wraps to (Segoe UI averages ~0.52 em per character)."""
    chars_per_line = max(1, int(width_in * 72 / (size_pt * 0.52)))
    lines, current = 1, 0
    for word in str(text).split():
        if current and current + 1 + len(word) > chars_per_line:
            lines, current = lines + 1, len(word)
        else:
            current += (1 if current else 0) + len(word)
    return lines


_FONT_FILES = {  # Segoe UI on Windows first; similar-width fonts elsewhere
    True: ["C:/Windows/Fonts/segoeuib.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
           "C:/Windows/Fonts/arialbd.ttf", "/Library/Fonts/Arial Bold.ttf"],
    False: ["C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "C:/Windows/Fonts/arial.ttf", "/Library/Fonts/Arial.ttf"],
}
_font_cache = {}


def _fits(text, size_pt, width_in, bold=False) -> bool:
    """True if text fits on one line at this size: measured with the real font when possible."""
    if bold not in _font_cache:
        _font_cache[bold] = None
        try:
            from PIL import ImageFont
            import os
            for path in _FONT_FILES[bold]:
                if os.path.exists(path):
                    _font_cache[bold] = ImageFont.truetype(path, 100)
                    break
        except ImportError:
            pass
    font = _font_cache[bold]
    em = font.getlength(str(text)) / 100 if font else len(str(text)) * 0.62
    return em * size_pt * 1.06 <= width_in * 72      # 6% safety margin


def _text(frame, text, size, color=jt.INK, bold=False, align=PP_ALIGN.LEFT):
    frame.clear()
    frame.word_wrap = True
    p = frame.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = str(text)
    run.font.name, run.font.size, run.font.bold = FONT, Pt(size), bold
    run.font.color.rgb = _rgb(color)
    return p


class Deck:
    """A 16:9 deck that starts with a title slide."""

    def __init__(self, title: str, subtitle: str = "", source: str = ""):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.blank = self.prs.slide_layouts[6]
        self.source = source
        self._title_slide(title, subtitle)

    # ------------------------------------------------------------------ basics
    def _slide(self):
        slide = self.prs.slides.add_slide(self.blank)
        bg = slide.background.fill
        bg.solid()
        bg.fore_color.rgb = _rgb(jt.PAGE)
        return slide

    def _box(self, slide, left, top, width, height):
        return slide.shapes.add_textbox(left, top, width, height).text_frame

    def _header(self, slide, title, takeaway=""):
        """Slide title (the takeaway headline), a short accent rule, and an optional explanation.

        Everything below the title is positioned from the title's estimated height, so a
        long title that wraps pushes the rule and takeaway down instead of overlapping them.
        """
        width_in = (W - 2 * MARGIN) / 914400
        size = 26 if len(title) <= 60 else 22 if len(title) <= 90 else 20
        title_lines = _lines(title, size, width_in)
        title_h = Pt(size * 1.2 * title_lines + 6)
        frame = self._box(slide, MARGIN, Inches(0.4), W - 2 * MARGIN, title_h)
        _text(frame, title, size, bold=True)
        frame.vertical_anchor = MSO_ANCHOR.TOP

        rule_top = Inches(0.4) + title_h + Inches(0.1)
        rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, rule_top, Inches(0.9), Emu(38100))
        rule.fill.solid()
        rule.fill.fore_color.rgb = _rgb(jt.PALETTE[0])
        rule.line.fill.background()
        top = rule_top + Inches(0.2)
        if takeaway:
            take_h = Pt(15 * 1.3 * _lines(takeaway, 15, width_in) + 6)
            t = self._box(slide, MARGIN, top, W - 2 * MARGIN, take_h)
            _text(t, takeaway, 15, color=jt.INK_2)
            top = top + take_h + Inches(0.15)
        self._footer(slide)
        return top

    def _footer(self, slide):
        n = len(self.prs.slides)
        frame = self._box(slide, MARGIN, H - Inches(0.5), W - 2 * MARGIN, Inches(0.3))
        label = f"{self.source}   ·   {n}" if self.source else str(n)
        _text(frame, label, 10, color=jt.MUTED, align=PP_ALIGN.RIGHT)

    def _title_slide(self, title, subtitle):
        slide = self._slide()
        band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.35), H)
        band.fill.solid()
        band.fill.fore_color.rgb = _rgb(jt.PALETTE[0])
        band.line.fill.background()
        frame = self._box(slide, Inches(1.1), Inches(2.5), Inches(11), Inches(1.4))
        _text(frame, title, 40, bold=True)
        if subtitle:
            sub = self._box(slide, Inches(1.1), Inches(3.9), Inches(11), Inches(1.0))
            _text(sub, subtitle, 18, color=jt.INK_2)
        by = self._box(slide, Inches(1.1), H - Inches(1.2), Inches(11), Inches(0.4))
        _text(by, "Prepared by Marro Briggs", 12, color=jt.MUTED)

    # ------------------------------------------------------------------ slides
    def kpis(self, title, items, takeaway=""):
        """3-4 headline numbers as cards: [(label, value), ...] or [(label, value, note), ...].

        Keep each value to one short number ("+54%", "$4.0M"); comparisons go in the note.
        All values share one font size, shrunk only as far as the longest value needs.
        """
        slide = self._slide()
        top = self._header(slide, title, takeaway)
        items = list(items)[:4]
        gap = Inches(0.3)
        width = int((W - 2 * MARGIN - gap * (len(items) - 1)) / len(items))
        inner_in = (width - Inches(0.6)) / 914400
        # One value size for every card: the largest that keeps the longest value on one line.
        value_size = 34
        while value_size > 16 and not all(_fits(it[1], value_size, inner_in - 0.15, bold=True) for it in items):
            value_size -= 2
        for i, item in enumerate(items):
            left = MARGIN + i * (width + gap)
            card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top + Inches(0.4), width, Inches(2.2))
            card.adjustments[0] = 0.08
            card.fill.solid()
            card.fill.fore_color.rgb = _rgb(jt.SURFACE)
            card.line.color.rgb = _rgb(jt.GRID)
            card.shadow.inherit = False
            lab = self._box(slide, left + Inches(0.3), top + Inches(0.6), width - Inches(0.6), Inches(0.55))
            _text(lab, item[0], 13 if _lines(item[0], 14, inner_in) > 1 else 14, color=jt.INK_2)
            val = self._box(slide, left + Inches(0.3), top + Inches(1.15), width - Inches(0.6), Inches(0.8))
            _text(val, item[1], value_size, bold=True)
            if len(item) > 2 and item[2]:
                note = self._box(slide, left + Inches(0.3), top + Inches(1.9), width - Inches(0.6), Inches(0.5))
                _text(note, item[2], 11, color=jt.MUTED)
        return slide

    def _chart(self, slide, top, chart_type, categories, series: dict):
        data = CategoryChartData()
        data.categories = [str(c) for c in categories]
        for name, values in series.items():
            data.add_series(str(name), [float(v) for v in values])
        height = H - top - Inches(0.75)
        frame = slide.shapes.add_chart(chart_type, MARGIN, top, W - 2 * MARGIN, height, data)
        chart = frame.chart
        chart.has_title = False          # the slide headline is the title
        chart.font.name, chart.font.size = FONT, Pt(12)
        chart.font.color.rgb = _rgb(jt.INK_2)
        return chart

    def _style_axes(self, chart, number_format, value_title, category_title, gridlines=True,
                    hide_values=False):
        va, ca = chart.value_axis, chart.category_axis
        if hide_values:                  # bars carry their own labels; a value axis would repeat them
            va.visible = False
            gridlines = False
            value_title = ""
        n = len(chart.plots[0].categories)
        if n > 12:                       # thin out crowded category labels (e.g. many months)
            from pptx.oxml.ns import qn
            from lxml import etree
            axis = ca._element
            for old_skip in axis.findall(qn("c:tickLblSkip")):
                axis.remove(old_skip)
            skip = etree.Element(qn("c:tickLblSkip"))
            skip.set("val", str(-(-n // 12)))
            # PowerPoint requires a fixed child order: tickLblSkip goes before these
            after = [axis.find(qn(t)) for t in ("c:tickMarkSkip", "c:noMultiLvlLbl", "c:extLst")]
            after = [a for a in after if a is not None]
            if after:
                after[0].addprevious(skip)
            else:
                axis.append(skip)
        va.has_major_gridlines = gridlines
        if gridlines:
            va.major_gridlines.format.line.color.rgb = _rgb(jt.GRID)
        va.format.line.fill.background()
        va.tick_labels.number_format = number_format
        va.tick_labels.number_format_is_linked = False
        va.tick_labels.font.color.rgb = _rgb(jt.MUTED)
        ca.format.line.color.rgb = _rgb(jt.BASELINE)
        ca.tick_labels.font.color.rgb = _rgb(jt.INK_2)
        ca.has_major_gridlines = False
        for axis, label in ((va, value_title), (ca, category_title)):
            if label:
                axis.has_title = True
                _text(axis.axis_title.text_frame, label, 12, color=jt.INK_2)

    def bar(self, title, categories, values, takeaway="", number_format="#,##0",
            value_title="", category_title="", highlight=None, horizontal=True, labels=True):
        """One-series ranked bars. highlight = a category (or list) to show in blue, the rest gray.

        Pass categories already sorted (largest first). At most ~12 bars.
        """
        slide = self._slide()
        top = self._header(slide, title, takeaway)
        kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
        chart = self._chart(slide, top, kind, categories, {value_title or "Value": values})
        chart.has_legend = False
        plot = chart.plots[0]
        plot.gap_width = 60
        series = plot.series[0]
        series.format.fill.solid()
        series.format.fill.fore_color.rgb = _rgb(jt.PALETTE[0])
        if highlight is not None:
            focus = {highlight} if isinstance(highlight, str) else set(highlight)
            for i, cat in enumerate(categories):
                point = series.points[i]
                point.format.fill.solid()
                point.format.fill.fore_color.rgb = _rgb(jt.PALETTE[0] if cat in focus else jt.DIMMED)
        if labels:
            plot.has_data_labels = True
            dl = plot.data_labels
            dl.number_format, dl.number_format_is_linked = number_format, False
            dl.position = XL_LABEL_POSITION.OUTSIDE_END
            dl.font.size, dl.font.color.rgb = Pt(11), _rgb(jt.INK_2)
        if horizontal:
            chart.category_axis.reverse_order = True   # largest at the top
        self._style_axes(chart, number_format, value_title, category_title, hide_values=labels)
        return slide

    def grouped_bar(self, title, categories, series: dict, takeaway="", number_format="#,##0",
                    value_title="", category_title="", horizontal=False):
        """Side-by-side bars for 2-4 measures per category, e.g. {"Revenue": [...], "Profit": [...]}."""
        slide = self._slide()
        top = self._header(slide, title, takeaway)
        kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
        chart = self._chart(slide, top, kind, categories, series)
        chart.has_legend = True
        chart.legend.position, chart.legend.include_in_layout = XL_LEGEND_POSITION.TOP, False
        chart.legend.font.size = Pt(12)
        plot = chart.plots[0]
        plot.gap_width, plot.overlap = 70, -10
        for i, s in enumerate(plot.series):
            s.format.fill.solid()
            s.format.fill.fore_color.rgb = _rgb(jt.PALETTE[i % len(jt.PALETTE)])
        if horizontal:
            chart.category_axis.reverse_order = True
        self._style_axes(chart, number_format, value_title, category_title)
        return slide

    def line(self, title, categories, series: dict, takeaway="", number_format="#,##0",
             value_title="", category_title=""):
        """Trend over time. categories = period labels in order; series = {name: values}."""
        slide = self._slide()
        top = self._header(slide, title, takeaway)
        chart = self._chart(slide, top, XL_CHART_TYPE.LINE_MARKERS, categories, series)
        chart.has_legend = len(series) > 1
        if chart.has_legend:
            chart.legend.position, chart.legend.include_in_layout = XL_LEGEND_POSITION.TOP, False
        for i, s in enumerate(chart.plots[0].series):
            color = _rgb(jt.PALETTE[i % len(jt.PALETTE)])
            s.format.line.color.rgb, s.format.line.width = color, Pt(2.25)
            s.smooth = False
            s.marker.size = 7
            s.marker.format.fill.solid()
            s.marker.format.fill.fore_color.rgb = color
            s.marker.format.line.color.rgb = _rgb(jt.SURFACE)
        self._style_axes(chart, number_format, value_title, category_title)
        return slide

    def table(self, title, rows, takeaway="", columns=None, max_rows=12):
        """A clean table. rows = a pandas DataFrame, or a list of lists with columns=[...]."""
        slide = self._slide()
        top = self._header(slide, title, takeaway)
        if hasattr(rows, "columns"):
            columns = [str(c) for c in rows.columns]
            rows = rows.astype(str).values.tolist()
        rows = [[str(v) for v in r] for r in list(rows)[:max_rows]]
        n_rows, n_cols = len(rows) + 1, len(columns)
        row_h = Inches(0.38)
        shape = slide.shapes.add_table(n_rows, n_cols, MARGIN, top + Inches(0.1),
                                       W - 2 * MARGIN, row_h * n_rows)
        table = shape.table
        for c, name in enumerate(columns):
            cell = table.cell(0, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = _rgb(jt.PALETTE[0])
            _text(cell.text_frame, name, 12, color="#ffffff", bold=True)
        for r, row in enumerate(rows, start=1):
            for c, value in enumerate(row):
                cell = table.cell(r, c)
                cell.fill.solid()
                cell.fill.fore_color.rgb = _rgb(jt.SURFACE if r % 2 else "#f0efec")
                is_number = value.replace(",", "").replace("$", "").replace("%", "").replace(".", "", 1).lstrip("-").isdigit()
                _text(cell.text_frame, value, 12, color=jt.INK,
                      align=PP_ALIGN.RIGHT if is_number else PP_ALIGN.LEFT)
        return slide

    def bullets(self, title, items, takeaway=""):
        """Recommendations, next steps or key points: 2-5 short items."""
        slide = self._slide()
        top = self._header(slide, title, takeaway)
        for i, item in enumerate(list(items)[:5]):
            y = top + Inches(0.25) + i * Inches(0.95)
            num = slide.shapes.add_shape(MSO_SHAPE.OVAL, MARGIN, y, Inches(0.5), Inches(0.5))
            num.fill.solid()
            num.fill.fore_color.rgb = _rgb(jt.PALETTE[0])
            num.line.fill.background()
            _text(num.text_frame, i + 1, 14, color="#ffffff", bold=True, align=PP_ALIGN.CENTER)
            num.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
            frame = self._box(slide, MARGIN + Inches(0.75), y - Inches(0.02), W - 2 * MARGIN - Inches(0.75), Inches(0.85))
            _text(frame, item, 17)
        return slide

    def caveats(self, items, title="How to read these numbers"):
        """The limits of the analysis, so nobody presents a number further than it goes."""
        slide = self._slide()
        top = self._header(slide, title)
        frame = self._box(slide, MARGIN, top + Inches(0.1), W - 2 * MARGIN, H - top - Inches(1))
        frame.word_wrap = True
        for i, item in enumerate(items):
            p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            p.space_after = Pt(10)
            run = p.add_run()
            run.text = f"•  {item}"
            run.font.name, run.font.size = FONT, Pt(16)
            run.font.color.rgb = _rgb(jt.INK_2)
        return slide

    def save(self, path="outputs/report.pptx"):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.prs.save(path)
        print(f"Saved deck: {path} ({len(self.prs.slides)} slides)")
        return path
