"""Helpers for interactive figures (design D21). Imported by figure scripts with
    import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _interactive import plotly_html, threejs_html
`plotly_html(path, traces, layout)` writes a self-contained HTML page loading Plotly.js from its CDN with the given
JSON traces/layout; `threejs_html(path, scene)` saves a Sage 3D scene with the threejs viewer (CDN build).
Outputs go to figures/interactive/<name>.html and are embedded by the `interactive` shortcode.
"""
import json, os

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interactive")
PLOTLY = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _f(x):
    """Convert Sage/numpy numbers to plain floats for JSON."""
    try:
        return float(x)
    except Exception:
        return x


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, str):
        return obj
    return _f(obj)


def plotly_html(name, traces, layout=None, config=None):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name + ".html")
    layout = dict(layout or {}); layout.setdefault("margin", {"l": 50, "r": 20, "t": 30, "b": 50})
    config = dict(config or {}); config.setdefault("responsive", True); config.setdefault("displaylogo", False)
    html = ("<!DOCTYPE html><html><head><meta charset='utf-8'><script src='%s'></script></head>"
            "<body style='margin:0'><div id='fig' style='width:100%%;height:100vh'></div><script>"
            "Plotly.newPlot('fig', %s, %s, %s);</script></body></html>") % (
        PLOTLY, json.dumps(_clean(traces)), json.dumps(_clean(layout)), json.dumps(config))
    open(path, "w").write(html)
    return path


def threejs_html(name, scene, **kw):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name + ".html")
    scene.save(path, online=True, **kw)
    return path
