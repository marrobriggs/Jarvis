"""
jarvis_theme: one look for every Jarvis report.

Jarvis copies this file into each job's work folder. The analyst's code
imports it, so every chart and every report app shares the same colors,
fonts and layout, no matter which job or which model wrote the code.

The color palette is a validated, colorblind-safe set of 8 hues used in a
FIXED order. Color follows the thing it represents (the West region is
always the same color), never its rank.

Usage in a Streamlit report (app.py):
    import jarvis_theme as jt
    jt.page("Pizza Sales Review", "Full-year 2015 orders, 4 locations")
    jt.kpis([("Revenue", jt.money(817860)), ("Orders", "21,350")])
    jt.section("When are we busiest?")
    jt.chart(fig, "Friday and Saturday dinner drive 40% of orders.", table=df)
    jt.footer("Source: client order data, 2015", ["One year of data only."])
"""

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

# Categorical colors, in fixed order. Validated for colorblind safety on the
# light surface below. Never cycle past 8: fold extra categories into "Other".
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]

# One hue, light to dark, for magnitude (heatmaps, intensity).
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

# Two opposite hues with a neutral middle, for above/below comparisons.
DIVERGING = ["#2a78d6", "#86b6ef", "#f0efec", "#ec9a99", "#e34948"]

# Reserved for meaning good/bad. Never use these for ordinary categories.
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

SURFACE = "#fcfcfb"      # chart and card background
PAGE = "#f9f9f7"         # app background
INK = "#0b0b0b"          # primary text
INK_2 = "#52514e"        # secondary text
MUTED = "#898781"        # axis labels, de-emphasized marks
GRID = "#e1e0d9"         # gridlines
BASELINE = "#c3c2b7"     # axis lines
DIMMED = "#c3c2b7"       # non-highlighted bars in an emphasis chart
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def color_map(categories) -> dict:
    """Give each category a fixed color, in the order given.

    Pass categories in a stable order (for example sorted by name, or the
    natural order of regions) so the same category gets the same color in
    every chart. More than 8 categories: fold the rest into "Other" first.
    """
    cats = list(dict.fromkeys(categories))
    mapping = {c: PALETTE[i] for i, c in enumerate(cats[:8])}
    for c in cats[8:]:
        mapping[c] = MUTED  # beyond 8 hues colors stop being distinguishable
    if "Other" in mapping:
        mapping["Other"] = MUTED
    return mapping


def highlight(categories, focus) -> dict:
    """Emphasis colors: the focus category(s) in blue, everything else gray."""
    focus = {focus} if isinstance(focus, str) else set(focus)
    return {c: (PALETTE[0] if c in focus else DIMMED) for c in categories}


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

def money(x, symbol: str = "$", decimals: int = 0) -> str:
    """1234567 -> '$1.23M', 45210 -> '$45.2K', 950 -> '$950'."""
    x = float(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000:
        return f"{sign}{symbol}{x / 1_000_000:.2f}M"
    if x >= 10_000:
        return f"{sign}{symbol}{x / 1_000:.1f}K"
    return f"{sign}{symbol}{x:,.{decimals}f}"


def pct(x, decimals: int = 1) -> str:
    """0.1234 -> '12.3%'. Pass a fraction, not a percent."""
    return f"{float(x) * 100:.{decimals}f}%"


def num(x) -> str:
    """12345.6 -> '12,346'."""
    return f"{float(x):,.0f}"


# ---------------------------------------------------------------------------
# Plotly template: applied automatically on import
# ---------------------------------------------------------------------------

def _register_plotly_template() -> None:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        return  # plotly not installed; nothing to register

    axis = dict(
        showgrid=True, gridcolor=GRID, gridwidth=1,
        showline=True, linecolor=BASELINE, linewidth=1,
        zeroline=False, ticks="", automargin=True,
        tickfont=dict(color=MUTED, size=12),
        title=dict(font=dict(color=INK_2, size=13)),
    )
    layout = go.Layout(
        font=dict(family=FONT, color=INK, size=13),
        title=dict(font=dict(size=16, color=INK), x=0, xanchor="left", y=0.97),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        colorway=PALETTE,
        colorscale=dict(sequential=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
                        diverging=[[i / (len(DIVERGING) - 1), c] for i, c in enumerate(DIVERGING)]),
        xaxis=axis, yaxis=dict(axis, showline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    title=dict(text=""), font=dict(color=INK_2, size=12)),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font=dict(family=FONT, color=INK, size=12)),
        hovermode="closest",
        margin=dict(l=56, r=24, t=64, b=48),
        bargap=0.3, bargroupgap=0.08,
    )
    template = go.layout.Template(layout=layout)
    template.data.scatter = [go.Scatter(line=dict(width=2), marker=dict(size=8))]
    template.data.bar = [go.Bar(marker=dict(line=dict(width=0)))]
    try:
        template.layout.barcornerradius = 4  # rounded bar ends (newer Plotly only)
    except (ValueError, AttributeError):
        pass
    pio.templates["jarvis"] = template
    pio.templates.default = "jarvis"


try:
    _register_plotly_template()
except Exception as _e:  # never let styling break a job; charts just fall back to Plotly's look
    print(f"jarvis_theme: Plotly styling unavailable ({type(_e).__name__}: {_e})")


# ---------------------------------------------------------------------------
# Streamlit report building blocks
# ---------------------------------------------------------------------------

_CSS = f"""
<style>
  .block-container {{ max-width: 1150px; padding-top: 2rem; padding-bottom: 3rem; }}
  h1 {{ font-weight: 650; letter-spacing: -0.01em; }}
  [data-testid="stMetric"] {{
      background: {SURFACE}; border: 1px solid rgba(11,11,11,0.10);
      border-radius: 12px; padding: 14px 16px;
  }}
  [data-testid="stMetricLabel"] p {{ color: {INK_2}; font-size: 0.85rem; }}
  [data-testid="stMetricValue"] {{ font-weight: 650; }}
  .jt-insight {{ color: {INK_2}; font-size: 0.95rem; margin: -0.25rem 0 0.75rem 0; }}
  .jt-footer {{ color: {MUTED}; font-size: 0.8rem; }}
  [data-testid="stPlotlyChart"] {{
      border: 1px solid rgba(11,11,11,0.10); border-radius: 12px; overflow: hidden;
  }}
</style>
"""


def page(title: str, subtitle: str = "") -> None:
    """Start the report. Call this first, once."""
    import streamlit as st
    st.set_page_config(page_title=title, layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def kpis(items) -> None:
    """A row of headline numbers: [(label, value), (label, value, delta), ...]. 3-4 is ideal."""
    import streamlit as st
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value = item[0], item[1]
        delta = item[2] if len(item) > 2 else None
        col.metric(label, value, delta)


def section(title: str, intro: str = "") -> None:
    """A titled section. Phrase the title as the question it answers."""
    import streamlit as st
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.subheader(title)
    if intro:
        st.markdown(f'<p class="jt-insight">{intro}</p>', unsafe_allow_html=True)


def chart(fig, insight: str = "", table=None) -> None:
    """Show a Plotly chart in the Jarvis style, with a one-line takeaway and a data table."""
    import plotly.io as pio
    import streamlit as st
    import inspect
    fig.update_layout(template=pio.templates["jarvis"])
    # theme=None keeps Streamlit from repainting the chart with its own colors
    if "width" in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(fig, theme=None, width="stretch")       # newer Streamlit
    else:
        st.plotly_chart(fig, theme=None, use_container_width=True)
    if insight:
        st.markdown(f'<p class="jt-insight">{insight}</p>', unsafe_allow_html=True)
    if table is not None:
        with st.expander("View data"):
            st.dataframe(table, hide_index=True)


def bullets(title: str, items) -> None:
    """A short titled list, for recommendations or next steps."""
    import streamlit as st
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.subheader(title)
    st.markdown("\n".join(f"- {i}" for i in items))


def footer(source: str, caveats=()) -> None:
    """Data source and caveats, quietly at the bottom."""
    import streamlit as st
    st.divider()
    if caveats:
        st.markdown("**Notes and limits**")
        st.markdown("\n".join(f"- {c}" for c in caveats))
    st.markdown(f'<p class="jt-footer">{source} · Prepared by Marro Briggs</p>', unsafe_allow_html=True)


STREAMLIT_CONFIG = f"""[theme]
base = "light"
primaryColor = "{PALETTE[0]}"
backgroundColor = "{PAGE}"
secondaryBackgroundColor = "#f0efec"
textColor = "{INK}"
font = "sans serif"

[client]
toolbarMode = "minimal"
"""
