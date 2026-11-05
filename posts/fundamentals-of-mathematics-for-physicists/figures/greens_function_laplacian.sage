# Figure: Green's functions. Left: the fundamental solution -1/(4 pi r) of the Laplacian in three dimensions (black)
# and the Yukawa Green's functions e^(-mr)/(4 pi r) of -Laplacian + m^2 for m = 1, 2 (blue, red), the mass shortening
# the range to 1/m. Right: the Green's function G(x, xi) of -d^2/dx^2 with Dirichlet conditions on [0, 1] for three
# source points xi: a tent with a kink of slope jump -1 at the source. The [PASS] lines verify the defining
# properties (harmonic away from the source, unit flux through every sphere, slope jump) and the momentum-space form
# 1/(k^2 + m^2) of the Yukawa Green's function by exact symbolic computation. Output:
# figures/greens_function_laplacian.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "greens_function_laplacian.svg")
x, y, z, r, R, m, k, xi, th, ph, s = var("x y z r R m k xi th ph s", domain="real")
assume(R > 0, m > 0, r > 0)
rr = sqrt(x ** 2 + y ** 2 + z ** 2)
G3 = -1 / (4 * pi * rr)
lap = lambda u: diff(u, x, 2) + diff(u, y, 2) + diff(u, z, 2)
print("[PASS] G = -1/(4 pi r) is harmonic away from the origin in three dimensions:", bool(lap(G3).simplify_full() == 0))
# flux of grad G through the sphere of radius R: grad(-1/(4 pi r)) = r_hat / (4 pi r^2), times the area 4 pi R^2
gradG_radial = diff(-1 / (4 * pi * r), r)
flux3 = integrate(integrate(gradG_radial.subs(r=R) * R ** 2 * sin(th), th, 0, pi), ph, 0, 2 * pi)
print("[PASS] the flux of grad G through every sphere around the origin is 1, so Laplacian G = delta:", bool(flux3.simplify_full() == 1))
G2 = log(x ** 2 + y ** 2) / (4 * pi)
flux2 = integrate(diff(log(r) / (2 * pi), r).subs(r=R) * R, ph, 0, 2 * pi)
print("[PASS] in two dimensions G = log(r) / (2 pi) is harmonic away from the origin with unit flux through every circle:",
      bool((diff(G2, x, 2) + diff(G2, y, 2)).simplify_full() == 0) and bool(flux2.simplify_full() == 1))
Y = exp(-m * rr) / (4 * pi * rr)
print("[PASS] the Yukawa function e^(-mr)/(4 pi r) satisfies (-Laplacian + m^2) G = 0 away from the origin:",
      bool((-lap(Y) + m ** 2 * Y).simplify_full() == 0))
Yr = exp(-m * r) / (4 * pi * r)
fluxY = (-diff(Yr, r) * 4 * pi * r ** 2).simplify_full()
print("[PASS] the inward flux of grad(e^(-mr)/(4 pi r)) through the sphere of radius R is e^(-mR)(1 + mR), tending to 1 as R -> 0:",
      bool((fluxY - exp(-m * r) * (1 + m * r)).simplify_full() == 0) and bool(limit(fluxY, r=0) == 1))
assume(r > 0)
ft = integrate(k * sin(k * r) / (k ** 2 + m ** 2), k, 0, oo)
print("[PASS] the Fourier integral (2 pi^2 r)^(-1) int_0^oo k sin(kr)/(k^2 + m^2) dk of 1/(k^2 + m^2) equals e^(-mr)/(4 pi r):",
      bool((ft / (2 * pi ** 2 * r) - Yr).simplify_full() == 0))
G1 = lambda xx, s: xx * (1 - s) if xx <= s else s * (1 - xx)
Ga, Gb = x * (1 - xi), xi * (1 - x)        # G(x, xi) for x < xi and x > xi
u1 = integrate(Gb.subs(xi=s), s, 0, x).subs(s=xi) + integrate(Ga.subs(xi=s), s, x, 1)   # u = int G(x, xi) f(xi) dxi with f = 1
print("[PASS] the Dirichlet Green's function of -d^2/dx^2 on [0, 1] has slope jump -1 at x = xi and gives u = x(1 - x)/2 for f = 1:",
      bool((diff(Gb, x) - diff(Ga, x)).subs(x=xi) == -1) and bool(Ga.subs(x=xi) == Gb.subs(x=xi)) and bool((u1 - x * (1 - x) / 2).expand() == 0)
      and bool(-diff(x * (1 - x) / 2, x, 2) == 1) and bool(Ga.subs(x=0) == 0) and bool(Gb.subs(x=1) == 0))
print("[PASS] the Green's function is symmetric, G(x, xi) = G(xi, x):", bool(Ga.subs({x: xi, xi: x}) == Gb))
left = plot(-1 / (4 * pi * r), (r, 0.08, 3), color="black", thickness=1.5, legend_label=r"$-1/(4\pi r)$: $\Delta G = \delta$")
for mv, col in [(1, "royalblue"), (2, "firebrick")]:
    left += plot(-exp(-mv * r) / (4 * pi * r), (r, 0.08, 3), color=col, thickness=1.5, legend_label=r"$-e^{-%sr}/(4\pi r)$: $(\Delta - %d) G = \delta$" % ("" if mv == 1 else mv, mv ** 2))
left += text(r"range $1/m$ of the Yukawa Green's function", (1.9, -0.55), fontsize=9)
left.axes_labels([r"$r$", r"$G(r)$"]); left.set_legend_options(loc="lower right", font_size=8)
left.ymin(-1.2); left.ymax(0.05)
right = Graphics()
for s0, col in [(0.3, "royalblue"), (0.5, "darkgreen"), (0.7, "firebrick")]:
    right += plot(lambda xx: G1(xx, s0), (0, 1), color=col, thickness=1.5, legend_label=r"$\xi = %.1f$" % s0)
    right += point((s0, s0 * (1 - s0)), color=col, size=25, zorder=5)
right += text(r"$G(x,\xi)$ of $-d^2/dx^2$ on $[0,1]$, $G(0,\xi)=G(1,\xi)=0$", (0.5, 0.29), fontsize=9)
right.axes_labels([r"$x$", r"$G(x,\xi)$"]); right.set_legend_options(loc="lower right", font_size=8)
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[10, 3.9]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

# interactive version: the three radial Green's functions with hover values
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
import math
rs = [0.08 + 2.92 * i / 300.0 for i in range(301)]
traces = [{"x": rs, "y": [-1 / (4 * math.pi * u) for u in rs], "mode": "lines", "line": {"color": "black", "width": 2}, "name": "-1/(4 pi r) (Laplacian)"}]
for mv, col in [(1, "royalblue"), (2, "firebrick")]:
    traces.append({"x": rs, "y": [-math.exp(-mv * u) / (4 * math.pi * u) for u in rs], "mode": "lines", "line": {"color": col, "width": 2}, "name": "-e^(-%d r)/(4 pi r) (Laplacian - %d)" % (mv, mv ** 2)})
plotly_html("greens_function_laplacian", traces, {"title": "Radial Green's functions in three dimensions (hover for values)", "xaxis": {"title": "r"}, "yaxis": {"title": "G(r)", "range": [-1.2, 0.05]}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/greens_function_laplacian.html")
