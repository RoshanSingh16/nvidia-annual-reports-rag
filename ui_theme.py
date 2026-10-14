"""Shared look & feel: CSS (Inter body, Plus Jakarta Sans headings) and a consistent Plotly template."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

ACCENT = "#76B900"
COLORS = ["#76B900", "#95B5D6", "#F5A524", "#B58BFF", "#2DD4BF", "#FF6B6B"]
BODY = "Inter, 'Segoe UI', system-ui, -apple-system, Roboto, Arial, sans-serif"
HEAD = "'Plus Jakarta Sans', Inter, 'Segoe UI', system-ui, sans-serif"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
:root {{
--bg:#0B0F14; --panel:#121820; --panel2:#171F29; --line:#223041; --text:#E6EDF3; --muted:#93A1B0;
--accent:{ACCENT}; --warn:#F5A524; --bad:#FF6B6B;
}}
html, body, [class*="css"], .stMarkdown, .stTextInput, .stChatInput, button, input, textarea {{
font-family:{BODY} !important; font-size:15px; line-height:1.6;
}}
h1,h2,h3,h4,.brand,.kpi .v,.page-title {{ font-family:{HEAD} !important; letter-spacing:-0.01em; }}
h1 {{ font-size:1.75rem !important; font-weight:700 !important; }}
h2 {{ font-size:1.3rem !important; font-weight:700 !important; margin-top:.4rem; }}
h3 {{ font-size:1.05rem !important; font-weight:600 !important; }}
.block-container {{ padding-top:1.6rem; padding-bottom:5rem; max-width:1180px; }}
section[data-testid="stSidebar"] {{ background:#F7F9F7; border-right:1px solid var(--line); }}

section[data-testid="stSidebar"] .block-container {{ padding-top:1rem; }}
.brand {{ display:flex; align-items:center; gap:.6rem; font-weight:800; font-size:1.05rem; margin-bottom:.2rem; }}
.brand .dot {{ width:10px; height:10px; border-radius:3px; background:var(--accent); box-shadow:0 0 12px {ACCENT}88; }}
.subtle {{ color:var(--muted); font-size:.85rem; }}
.page-title {{ font-size:1.75rem; font-weight:700; margin:0; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.75rem; margin:.8rem 0 1.2rem; }}
.kpi {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:.85rem 1rem; }}
.kpi .k {{ color:var(--muted); font-size:.74rem; text-transform:uppercase; letter-spacing:.06em; font-weight:500; }}
.kpi .v {{ font-size:1.55rem; font-weight:700;color:#FFFFFF; margin-top:.15rem; }}
.kpi .s {{ color:var(--muted); font-size:.75rem; }}
.chip {{ display:inline-block; padding:.12rem .55rem; border-radius:999px; font-size:.72rem; font-weight:600;
border:1px solid var(--line); color:var(--muted); background:var(--panel2); margin-right:.3rem; }}
.chip.ok {{ color:var(--accent); border-color:{ACCENT}55; }} .chip.warn {{ color:var(--warn); border-color:#F5A52455; }}
.chip.bad {{ color:var(--bad); border-color:#FF6B6B55; }}
.src {{ background:#F7F9F7; border:1px solid #386A00; border-left:3px solid var(--accent); border-radius:10px;
        padding:.7rem .9rem; margin:.45rem 0; }}
.src .t {{ font-weight:600; font-size:.9rem; }} .src .m {{ color:var(--muted); font-size:.78rem; margin:.1rem 0 .35rem; }}
.src pre {{ white-space:pre-wrap; word-break:break-word; font-family:'JetBrains Mono',ui-monospace,Menlo,Consolas,monospace;
            font-size:.8rem; line-height:1.5; margin:0; color:#C9D4DF; background:transparent; border:0; padding:0; }}
.callout {{ background:var(--panel2); border:1px solid var(--line); border-radius:10px; padding:.8rem 1rem; color:var(--muted); font-size:.88rem; }}
.callout b {{ color:var(--text); }}
[data-testid="stChatMessage"] {{ background:transparent; border:0; padding:.4rem 0; }}
[data-testid="stChatMessage"] table {{ font-size:.88rem; }}
[data-testid="stChatMessage"] code {{ font-size:.85em; }}
div[data-baseweb="tab-list"] {{ gap:.25rem; }}
.stButton>button, .stDownloadButton>button {{ border-radius:8px; border:1px solid var(--line); font-weight:500; }}
.stButton>button[kind="primary"] {{ background:var(--accent); color:#0B0F14; border:0; font-weight:600; }}
@media (max-width:760px) {{ .block-container {{ padding:1rem .8rem 5rem; }} .page-title {{ font-size:1.4rem; }} }}
</style>
"""


def kpi_html(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(f'<div class="kpi"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>' for k, v, s in items)
    return f'<div class="kpis">{cells}</div>'


def register_plotly_template() -> None:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=BODY, size=13, color="#C9D4DF"),
        title=dict(font=dict(family=HEAD, size=15, color="#E6EDF3"), x=0.01),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=COLORS,
        margin=dict(l=48, r=16, t=48, b=44),
        xaxis=dict(gridcolor="#1B2634", zerolinecolor="#223041", linecolor="#223041"),
        yaxis=dict(gridcolor="#1B2634", zerolinecolor="#223041", linecolor="#223041"),
        legend=dict(orientation="h", y=-0.22, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=dict(family=BODY, size=12), bgcolor="#121820", bordercolor="#223041"),
    )
    pio.templates["nv"] = t
    pio.templates.default = "nv"
