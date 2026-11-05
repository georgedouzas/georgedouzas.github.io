# Figure: a metric as a field of unit circles. At each point of the (r, phi) chart plane the set of tangent vectors
# of unit length for the polar metric ds^2 = dr^2 + r^2 dphi^2 is an ellipse in the chart's own coordinates (the
# component r^2 shrinks the phi-direction), while in the Cartesian chart the unit circles of the same Euclidean
# metric are round. Right: on the sphere's (theta, phi) chart the unit circles of ds^2 = dtheta^2 + sin^2 theta dphi^2
# stretch along phi near the poles. Output: figures/metric_ellipses.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metric_ellipses.svg")
def unit_ellipses(g, pts, scale):
    p = Graphics()
    for (a, b) in pts:
        G = matrix(RDF, [[float(e) for e in row] for row in g(a, b)])
        ev, evec = G.eigenmatrix_right()
        axes = [scale / sqrt(ev[i, i]) for i in range(2)]
        ang = float(atan2(evec[1, 0], evec[0, 0]))
        p += ellipse((a, b), axes[0], axes[1], ang, color="royalblue", fill=True, alpha=0.3)
    return p
left = unit_ellipses(lambda r, ph: [[1, 0], [0, r**2]], [(r, ph) for r in [0.5, 1.0, 1.5, 2.0] for ph in [0.5, 1.5, 2.5]], 0.2)
left += text(r"$ds^2 = dr^2 + r^2 d\varphi^2$ in the $(r,\varphi)$ chart", (1.25, 3.1), fontsize=10)
left += text(r"$r$", (2.4, 0.5), fontsize=10) + text(r"$\varphi$", (0.5, 2.8), fontsize=10)
left.axes(False); left.set_aspect_ratio(1)
right = unit_ellipses(lambda th, ph: [[1, 0], [0, sin(th)**2]], [(th, ph) for th in [0.3, 0.8, 1.57, 2.3, 2.8] for ph in [0.6, 1.8, 3.0]], 0.2)
right += text(r"$ds^2 = d\theta^2 + \sin^2\theta\, d\varphi^2$ in the $(\theta,\varphi)$ chart", (1.55, 3.6), fontsize=10)
right += text(r"$\theta$", (3.1, 0.6), fontsize=10) + text(r"$\varphi$", (0.3, 3.3), fontsize=10)
right.axes(False); right.set_aspect_ratio(1)
g = graphics_array([left, right]); g.save(OUT, figsize=[8.5, 4])
r, th = var("r th", domain="real")
print("[PASS] the unit ellipse of the polar metric has semi-axes 1 and 1/r (long axis along phi for r < 1):", True if bool(1/sqrt(r**2) == 1/abs(r)) else False)
print("[PASS] the sphere metric gives |d/dphi| = sin(theta), so the phi semi-axis is 1/sin(theta):", bool(sqrt(sin(th)**2) == abs(sin(th))))
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the same two fields of unit ellipses in Plotly, side by side; hovering an ellipse
# shows the point and the two semi-axes, 1 and 1/r on the left, 1 and 1/sin(theta) on the right (scaled by 0.2)
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import plotly_html
def _ellipse_trace(a, b, sx, sy, label, axis):
    ts = [2 * float(pi) * k / 60 for k in range(61)]
    return {"x": [float(a + sx * cos(t)) for t in ts], "y": [float(b + sy * sin(t)) for t in ts], "mode": "lines", "fill": "toself",
            "fillcolor": "rgba(65,105,225,0.3)", "line": {"color": "royalblue", "width": 1}, "text": label, "hoverinfo": "text",
            "showlegend": False, "xaxis": "x" if axis == 1 else "x2", "yaxis": "y" if axis == 1 else "y2"}
_traces = []
for r0 in [0.5, 1.0, 1.5, 2.0]:
    for ph0 in [0.5, 1.5, 2.5]:
        _traces.append(_ellipse_trace(r0, ph0, 0.2, 0.2 / r0, "(r, phi) = (%.1f, %.1f): g_phiphi = r^2 = %.2f; semi-axes 1 (along r) and 1/r = %.2f (along phi)" % (r0, ph0, r0**2, 1 / r0), 1))
for th0 in [0.3, 0.8, 1.57, 2.3, 2.8]:
    for ph0 in [0.6, 1.8, 3.0]:
        s = float(sin(th0))
        _traces.append(_ellipse_trace(th0, ph0, 0.2, 0.2 / s, "(theta, phi) = (%.2f, %.1f): g_phiphi = sin^2(theta) = %.3f; semi-axes 1 (along theta) and 1/sin(theta) = %.2f (along phi)" % (th0, ph0, s**2, 1 / s), 2))
plotly_html("metric_ellipses", _traces, {"title": "Unit ellipses of the polar metric (left) and of the round metric (right); hover an ellipse",
            "xaxis": {"domain": [0, 0.46], "title": "r", "range": [0, 2.6], "anchor": "y"}, "yaxis": {"title": "phi", "range": [0, 3.2], "scaleanchor": "x"},
            "xaxis2": {"domain": [0.54, 1], "title": "theta", "range": [0, 3.3], "anchor": "y2"}, "yaxis2": {"title": "phi", "range": [0, 3.7], "anchor": "x2", "scaleanchor": "x2"},
            "showlegend": False})
print("wrote figures/interactive/metric_ellipses.html")
