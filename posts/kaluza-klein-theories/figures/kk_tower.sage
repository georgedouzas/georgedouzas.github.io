# Figure: the Kaluza-Klein tower. Left: the masses |n|/R of the modes e^{i n y/R} for two radii, each level carrying
# the charge n e; the mode n is an eigenfunction of -d^2/dy^2 on the circle with eigenvalue (n/R)^2, which is checked
# symbolically, and periodicity y ~ y + 2 pi R requires e^{2 pi i n} = 1, i.e. n integer. Right: the first three
# mode functions (real parts). Output: figures/kk_tower.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kk_tower.svg")
yv, n, Rr = var("yv n Rr"); assume(Rr > 0)
mode = exp(I * n * yv / Rr)
print("[PASS] -d^2/dy^2 e^{i n y/R} = (n/R)^2 e^{i n y/R}:", bool((-diff(mode, yv, 2) - (n / Rr)^2 * mode).simplify_full() == 0))
print("[PASS] e^{i n (y + 2 pi R)/R} = e^{i n y/R} for integer n and not for n = 1/2:", bool(mode.subs(yv=yv + 2 * pi * Rr).subs(n=3).simplify_full() == mode.subs(n=3)) and not bool((mode.subs(yv=yv + 2 * pi * Rr) - mode).subs(n=1/2).subs(yv=0, Rr=1).simplify_full() == 0))
left = Graphics()
for (Rval, x0, col) in ((1.0, 0, "royalblue"), (2.0, 1, "firebrick")):
    for k in range(-4, 5):
        m = abs(k) / Rval
        left += line([(x0 + 0.1, m), (x0 + 0.9, m)], color=col, thickness=2)
        if k != 0: left += text("$n = \\pm%d$" % abs(k), (x0 + 0.5, m + 0.12), fontsize=7, color=col) if k > 0 else Graphics()
    left += text("$R = %.0f$" % Rval, (x0 + 0.5, 4.6), fontsize=10, color=col)
left += text("mass $|n|/R$, charge $n e$", (1, 5.1), fontsize=10)
left.axes_labels(["", "mass"]); left.xmin(-0.2); left.xmax(2.2); left.ymin(-0.3); left.ymax(5.4)
left.set_aspect_ratio("automatic")
right = Graphics()
for k, col in ((0, "gray"), (1, "royalblue"), (2, "firebrick")):
    right += plot(cos(k * yv), (yv, 0, 2 * pi), color=col, thickness=2, legend_label="$n = %d$" % k)
right.axes_labels(["$y/R$", "$\\mathrm{Re}\\,e^{iny/R}$"]); right.set_legend_options(loc="lower left", font_size=8)
fig = graphics_array([left, right]).matplotlib(figsize=[9, 3.8]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): a slider sets the radius R; the levels m_n = |n|/R of the tower move together, and
# the title counts the modes below the energy E = 2 (dashed line): 2 floor(E R) + 1, all of them light as R grows
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import plotly_html
import math as _m
_ns = list(range(-8, 9)); _Rs = [0.25 * k for k in range(1, 17)]; _E = 2.0
_traces = [{"x": [-8.5, 8.5], "y": [_E, _E], "mode": "lines", "name": "energy E = 2", "line": {"color": "gray", "dash": "dash"}}]
for Rv in _Rs:
    _traces.append({"x": _ns, "y": [abs(k) / Rv for k in _ns], "mode": "markers", "name": "m_n = |n|/R",
                    "marker": {"size": 11, "color": ["firebrick" if abs(k) / Rv < _E else "royalblue" for k in _ns]},
                    "text": ["n = %d, charge %d e, mass %.2f" % (k, k, abs(k) / Rv) for k in _ns], "hoverinfo": "text", "visible": abs(Rv - 1) < 1e-9})
_title = lambda Rv: "R = %.2f: %d modes lighter than E = 2 (red)" % (Rv, 2 * _m.floor(_E * Rv - 1e-12) + 1)
_steps = [{"method": "update", "label": "%.2f" % Rv, "args": [{"visible": [True] + [abs(Rv - Rw) < 1e-9 for Rw in _Rs]}, {"title": _title(Rv)}]} for Rv in _Rs]
plotly_html("kk_tower", _traces, {"title": _title(1.0), "xaxis": {"title": "mode number n (charge n e)"}, "yaxis": {"title": "mass m_n = |n|/R", "range": [-0.3, 12]}, "showlegend": False,
            "sliders": [{"active": 3, "currentvalue": {"prefix": "R = "}, "pad": {"t": 40}, "steps": _steps}]})
print("[PASS] the number of modes with |n|/R < E is 2 floor(E R) + 1 (E R not an integer), checked for E = 2 and R = 0.3, 1.3, 2.7:", all(sum(1 for k in range(-100, 101) if abs(k) / Rv < 2) == 2 * _m.floor(2 * Rv) + 1 for Rv in (0.3, 1.3, 2.7)))
print("wrote figures/interactive/kk_tower.html")
