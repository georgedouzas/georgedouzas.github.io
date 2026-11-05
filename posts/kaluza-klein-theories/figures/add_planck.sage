# Figure: large extra dimensions (Arkani-Hamed, Dimopoulos, Dvali). The size R of n extra dimensions needed for a
# fundamental scale M = 1, 10 TeV, from M_Pl^2 = M^{2+n} R^n (R^n the volume of the extra space) with
# M_Pl = G_N^{-1/2} = 1.22 10^19 GeV, against n = 1, ..., 7, with the scale of the sub-millimetre tests of Newton's law
# and the solar-system scale marked. Checks: the Gauss-law crossover of the potential from 1/r^{n+1} to 1/r at
# r ~ R reproduces the relation; R for n = 1 (of order 10^13 m, excluded) and n = 2 (millimetres); the spacing 1/R of
# the KK graviton tower; the identity (E R)^n/M_Pl^2 = E^n/M^{n+2} for the summed emission rate; and the KK sum
# over a torus reproducing the volume factor. Output: figures/add_planck.svg, interactive
# figures/interactive/add_planck.html.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "add_planck.svg")
MPl = 1.22e19                 # GeV
hbarc = 1.97327e-16           # GeV m
n, Ms, R, r, m1, m2, E = var("n M R r m1 m2 E", domain="positive")
def Rmetres(nn, M):
    return (MPl**2 / M**(nn + 2))**(1.0 / nn) * hbarc
# Gauss law: V(r) = m1 m2/(M^{n+2} r^{n+1}) for r << R; for r >> R the flux spreads over the volume R^n: m1 m2/(M^{n+2} R^n r)
Vshort = m1 * m2 / (Ms**(n + 2) * r**(n + 1)); Vlong = m1 * m2 / (Ms**(n + 2) * R**n * r)
print("[PASS] Gauss law: the short- and long-distance potentials agree at r = R, and matching Vlong to m1 m2/(M_Pl^2 r) gives M_Pl^2 = M^{n+2} R^n:",
      bool((Vshort - Vlong).subs(r=R).simplify_full() == 0) and bool(solve(Vlong == m1 * m2 / (var("MP")**2 * r), var("MP"))[-1].rhs()**2 == Ms**(n + 2) * R**n))
print("[PASS] n = 1, M = 1 TeV: R = %.1e m, of solar-system size, excluded:" % Rmetres(1, 1e3), 1e12 < Rmetres(1, 1e3) < 1e14)
print("[PASS] n = 2, M = 1 TeV: R = %.2f mm, the millimetre range of torsion-balance tests of Newton's law:" % (Rmetres(2, 1e3) * 1e3), 1e-4 < Rmetres(2, 1e3) < 1e-2)
print("[PASS] n = 6, M = 1 TeV: R = %.1e m, about 10^-14 m (1/R of a few MeV):" % Rmetres(6, 1e3), 1e-15 < Rmetres(6, 1e3) < 1e-13)
print("[PASS] R decreases with n for M = 1 TeV and with M for fixed n:",
      all(Rmetres(k, 1e3) > Rmetres(k + 1, 1e3) for k in range(1, 7)) and all(Rmetres(k, 1e3) > Rmetres(k, 1e4) for k in range(1, 8)))
gap_eV = 1 / (Rmetres(2, 1e3) / hbarc) * 1e9
print("[PASS] KK graviton spacing 1/R for n = 2, M = 1 TeV: %.1e eV, a nearly continuous tower:" % gap_eV, 1e-5 < gap_eV < 1e-3)
print("[PASS] summed emission rate: (E R)^n/M_Pl^2 = E^n/M^{n+2} with M_Pl^2 = M^{n+2} R^n:",
      bool(((E * R)**n / (Ms**(n + 2) * R**n) - E**n / Ms**(n + 2)).canonicalize_radical() == 0))
# KK reduction on a torus of side 2 pi R_c: the volume factor (2 pi R_c)^n multiplies M^{n+2}; number of KK modes below E
Rc = var("Rc", domain="positive")
cnt = lambda nn, ER: sum(1 for v in cartesian_product([range(-int(ER) - 1, int(ER) + 2)] * nn) if sum(x**2 for x in v) <= ER**2)
print("[PASS] KK modes on a torus of radius R_c below E: (E R_c)^n up to the ball volume factor (n = 2, E R_c = 30: %d vs pi 900 = %.0f):" % (cnt(2, 30), float(pi * 900)),
      abs(cnt(2, 30) / float(pi * 900) - 1) < 0.02)

ns = list(range(1, 8))
G = Graphics()
for M, c, lab in [(1e3, "firebrick", "$M = 1$ TeV"), (1e4, "royalblue", "$M = 10$ TeV")]:
    pts = [(k, log(Rmetres(k, M), 10)) for k in ns]
    G += line(pts, color=c, thickness=1.6, legend_label=lab) + point(pts, color=c, size=30)
G += line([(0.8, -4), (7.2, -4)], color="grey", linestyle="--", thickness=1)
G += text("sub-mm tests of Newton's law", (5.6, -3.2), color="grey", fontsize=9)
G += line([(0.8, 11), (7.2, 11)], color="grey", linestyle=":", thickness=1)
G += text("solar-system scale", (3.2, 11.8), color="grey", fontsize=9)
G.axes_labels(["$n$", "$\\log_{10}(R/\\mathrm{m})$"]); G.set_legend_options(loc="center right", font_size=9)
fig = G.matplotlib(figsize=[6.5, 4.2]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

# Interactive version: slide log10 of the fundamental scale M/GeV from 3 (1 TeV) to 6 (1000 TeV); the bars give
# log10(R/m) for n = 1..7 from M_Pl^2 = M^{n+2} R^n. Hover for values.
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json, math
def _data(lm):
    return [{"x": ns, "y": [math.log10(Rmetres(k, 10 ** lm)) for k in ns]}]
_frames, _steps = [], []
for i in range(0, 13):
    lm = 3 + 0.25 * i
    _frames.append({"name": "%.2f" % lm, "data": _data(lm), "traces": [0]})
    _steps.append({"label": "%.2f" % lm, "method": "animate", "args": [["%.2f" % lm], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]})
_traces = [dict(_data(3)[0], type="bar", name="log10(R/m)", marker={"color": "firebrick"}),
           {"x": [0.5, 7.5], "y": [-4, -4], "mode": "lines", "name": "0.1 mm", "line": {"color": "grey", "dash": "dash"}},
           {"x": [0.5, 7.5], "y": [-15, -15], "mode": "lines", "name": "1 fm", "line": {"color": "grey", "dash": "dot"}}]
_layout = {"title": "Size of n large extra dimensions: slide log10(M/GeV)",
           "xaxis": {"title": "number of extra dimensions n", "dtick": 1}, "yaxis": {"title": "log10(R / m)", "range": [-20, 14]},
           "sliders": [{"active": 0, "currentvalue": {"prefix": "log10(M/GeV) = "}, "steps": _steps, "pad": {"t": 30}}], "legend": {"orientation": "h", "y": -0.3}}
_path = plotly_html("add_planck", _traces, _layout)
_html = open(_path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(_frames)))
open(_path, "w").write(_html)
print("[PASS] slider: in every frame R^n M^{n+2} = M_Pl^2 for each n (relative error < 1e-9):",
      all(abs((10 ** fr["data"][0]["y"][j] / hbarc) ** ns[j] * (10 ** (3 + 0.25 * i)) ** (ns[j] + 2) / MPl**2 - 1) < 1e-9 for i, fr in enumerate(_frames) for j in range(7)))
print("wrote figures/interactive/add_planck.html")
