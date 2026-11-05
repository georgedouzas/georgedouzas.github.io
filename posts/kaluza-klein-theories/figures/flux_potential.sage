# Figure: moduli stabilisation by flux in a one-modulus toy model. Left: the four-dimensional potential of the radius
# R of an internal two-sphere threaded by N flux quanta, V(R) = a N^2/R^6 - b/R^4 (flux energy against internal
# curvature, in four-dimensional Einstein frame), for N = 1, 2, 3 with a = b = 1, and its minimum at
# R^2 = 3 a N^2/(2 b). Right: a two-flux landscape in the spirit of Bousso and Polchinski, the lattice of flux
# integers (n_1, n_2) with V = -V_0 + c_1 n_1^2 + c_2 n_2^2, and the thin shell |V| < epsilon of nearly vanishing
# vacuum energy. Checks: Dirac quantisation of the flux of a uniform field on the sphere (the patch potentials differ
# by a gauge transformation single valued exactly for integer N); the scalings of the two terms; the extremum, its
# value -4 b^3/(27 a^2 N^4) and its positive second derivative; the Bousso-Polchinski estimate pi^50/50! 10^100 ~ 10^60
# of lattice points in a 100-dimensional ball of radius 10. Output: figures/flux_potential.svg, interactive
# figures/interactive/flux_potential.html.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flux_potential.svg")
th, ph = var("theta phi", domain="real")
N, q, R, a, b = var("N q R a b", domain="positive")
# Dirac quantisation: a uniform field strength F = f vol_{S^2_R}, vol = R^2 sin(theta) dtheta dphi, with f = N/(2 q R^2)
f = N / (2 * q * R**2)
flux = integrate(integrate(f * R**2 * sin(th), th, 0, pi), ph, 0, 2 * pi)
print("[PASS] Dirac quantisation: F = N/(2 q R^2) vol on the sphere of radius R has flux int F = 2 pi N/q:", bool(flux == 2 * pi * N / q))
Aplus = (N / (2 * q)) * (1 - cos(th)); Aminus = -(N / (2 * q)) * (1 + cos(th))
print("[PASS] the patch potentials A_+- = (N/2q)(+-1 - cos theta) dphi both have dA = F and differ by (N/q) dphi:",
      bool(diff(Aplus, th) == f * R**2 * sin(th)) and bool(diff(Aminus, th) == f * R**2 * sin(th)) and bool(Aplus - Aminus == N / q))
print("[PASS] the transition function exp(i q (N/q) phi) is single valued on phi -> phi + 2 pi exactly for integer N:",
      all(bool(exp(I * n * 2 * pi) == 1) for n in range(-3, 4)) and not bool(exp(I * QQ(1/2) * 2 * pi) == 1))
# scalings: flux energy density |F|^2 ~ N^2/R^4 times internal area R^2, then the Einstein-frame factor R^{-4};
# curvature: -int sqrt(g) R_int = -(4 pi R^2)(2/R^2) = -8 pi, times R^{-4}
flux_term = (f**2 * 4 * pi * R**2 * R**(-4)).simplify_full()
curv_term = (-(4 * pi * R**2) * (2 / R**2) * R**(-4)).simplify_full()
print("[PASS] scalings: the flux term is proportional to N^2/R^6 and the curvature term to -1/R^4:",
      bool((flux_term * R**6 / N**2).simplify_full().has(R) == False) and bool((curv_term * R**4).simplify_full() == -8 * pi))
V = a * N**2 / R**6 - b / R**4
crit = solve(diff(V, R) == 0, R)
Rmin = sqrt(3 * a / (2 * b)) * N
print("[PASS] dV/dR = 0 at R^2 = 3 a N^2/(2 b): the flux fixes the radius, R_min grows linearly with N:",
      bool(diff(V, R).subs(R=Rmin).simplify_full() == 0) and any(bool((s.rhs() - Rmin).simplify_full() == 0) for s in crit))
Vmin = V.subs(R=Rmin).simplify_full()
print("[PASS] V(R_min) = -4 b^3/(27 a^2 N^4) < 0: the stabilised vacuum is anti-de Sitter:", bool((Vmin + 4 * b**3 / (27 * a**2 * N**4)).simplify_full() == 0))
V2 = diff(V, R, 2).subs(R=Rmin).simplify_full()
print("[PASS] V''(R_min) = 64 b^4/(27 a^3 N^6) > 0: a minimum, the modulus is massive:", bool((V2 - 64 * b**4 / (27 * a**3 * N**6)).simplify_full() == 0))
print("[PASS] V -> 0 from below as R -> infinity and V -> +infinity as R -> 0: the single critical point is the global minimum on R > 0:",
      bool(limit(V * R**4, R=oo) == -b) and bool(limit(V * R**6, R=0) == a * N**2))
# Bousso-Polchinski: lattice points in a ball of radius sqrt(L) = 10 in 100 dimensions ~ pi^50/50! 10^100
bp = pi**50 / factorial(50) * 10**100
print("[PASS] Bousso-Polchinski count: pi^50/50! * 10^100 = %.2e, of order 10^60:" % float(bp), 60 <= float(log(bp, 10)) < 61)
vol = lambda d, r: pi**(d / 2) / gamma(d / 2 + 1) * r**d
print("[PASS] the count is the volume pi^{d/2} r^d/Gamma(d/2 + 1) of the ball with d = 100, r = 10:", bool(vol(100, 10) == bp))

cols = {1: "royalblue", 2: "darkgreen", 3: "firebrick"}
left = Graphics()
for n, c in cols.items():
    Vn = V.subs(a=1, b=1, N=n)
    rmin = sqrt(1.5) * n
    left += plot(Vn * n**4, (R, 0.75 * n, 5 * n), color=c, thickness=1.8, legend_label="$N=%d$" % n)
    left += point((rmin, Vn.subs(R=rmin) * n**4), color=c, size=40)
left.axes_labels(["$R$", "$N^4\\,V(R)$"]); left.set_axes_range(ymin=-0.2, ymax=0.4); left.set_legend_options(loc="upper right", font_size=9)
right = Graphics()
V0, c1, c2, eps = 60.0, 1.0, float(sqrt(5)), 1.0
shell = []
for n1 in range(-9, 10):
    for n2 in range(-6, 7):
        Vv = -V0 + c1 * n1**2 + c2 * n2**2
        col = "firebrick" if abs(Vv) < eps else ("grey" if Vv < 0 else "lightgrey")
        right += point((n1, n2), color=col, size=30 if abs(Vv) < eps else 14)
        if abs(Vv) < eps: shell.append((n1, n2))
t = var("t")
for VV, ls in [(-eps, "--"), (eps, "--")]:
    right += parametric_plot((sqrt((V0 + VV) / c1) * cos(t), sqrt((V0 + VV) / c2) * sin(t)), (t, 0, 2 * pi), color="black", linestyle=ls, thickness=0.8)
print("[PASS] landscape panel: V = -60 + n_1^2 + sqrt(5) n_2^2 has exactly 8 flux vacua with |V| < 1 in the window shown:", len(shell) == 8)
right.axes_labels(["$n_1$", "$n_2$"]); right.set_aspect_ratio(1)
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[10, 4.2]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

# Interactive version: slide the number N of flux quanta (1 to 8). The curve is V(R) = N^2/R^6 - 1/R^4 itself, not
# rescaled, with its minimum marked: the radius grows like N and the depth falls like N^{-4}, so the axes rescale.
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json, math
def _data(n):
    rs = [0.8 * n + (5.2 * n) * i / 300.0 for i in range(301)]
    rm = math.sqrt(1.5) * n
    return [{"x": rs, "y": [n * n / r**6 - 1 / r**4 for r in rs]}, {"x": [rm], "y": [n * n / rm**6 - 1 / rm**4]}]
_frames, _steps = [], []
for n in range(1, 9):
    vmin = -4.0 / (27 * n**4)
    _frames.append({"name": str(n), "data": _data(n), "traces": [0, 1],
                    "layout": {"xaxis": {"range": [0, 6 * n], "title": "R"}, "yaxis": {"range": [2.5 * vmin, -4 * vmin], "title": "V(R)", "exponentformat": "e"}}})
    _steps.append({"label": str(n), "method": "animate", "args": [[str(n)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]})
_d = _data(1)
_traces = [dict(_d[0], mode="lines", name="V(R) = N^2/R^6 - 1/R^4", line={"color": "royalblue", "width": 3}),
           dict(_d[1], mode="markers", name="minimum R^2 = 3N^2/2, V = -4/(27 N^4)", marker={"color": "firebrick", "size": 11})]
_layout = {"title": "Flux stabilisation of the radius: slide the flux quanta N",
           "xaxis": {"range": [0, 6], "title": "R"}, "yaxis": {"range": [-10.0 / 27, 16.0 / 27], "title": "V(R)", "exponentformat": "e"},
           "sliders": [{"active": 0, "currentvalue": {"prefix": "N = "}, "steps": _steps, "pad": {"t": 30}}], "legend": {"orientation": "h", "y": -0.3}}
_path = plotly_html("flux_potential", _traces, _layout)
_html = open(_path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(_frames)))
open(_path, "w").write(_html)
print("[PASS] slider: every frame's marked point is the lowest sampled value of its curve:",
      all(min(fr["data"][0]["y"]) >= fr["data"][1]["y"][0] - 1e-15 for fr in _frames))
print("wrote figures/interactive/flux_potential.html")
