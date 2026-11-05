# Figure: the Randall-Sundrum warped geometry ds^2 = e^{-2k|y|} eta_{mu nu} dx^mu dx^nu - dy^2 in the signature
# (+, -, -, -, -). Left: the warp factor e^{-2k|y|} for k = 0.5, 1, 2, a sharp peak at the brane y = 0. Right: the
# graviton in the conformal coordinate z = sgn(y)(e^{k|y|} - 1)/k, the volcano potential 15k^2/(8(k|z| + 1)^2) of
# the analogue Schrodinger problem (with an attractive delta at z = 0, not drawn) and its normalised zero mode
# sqrt(k) (k|z| + 1)^{-3/2}, the localised four-dimensional graviton. Checks: the Einstein tensor of the warped metric
# for a general warp function A(y) computed from the Christoffel symbols with the book's curvature conventions; the
# bulk equations G_MN - Lambda g_MN = 0 force A'^2 = -Lambda/6 and A'' = 0, solved by A = k|y| away from y = 0 with
# Lambda = -6k^2; the jump A'' = 2k delta(y) is a brane of tension 6k/kappa_5^2 (24 M^3 k in Randall-Sundrum units);
# the Planck mass M_Pl^2 = (M^3/k)(1 - e^{-2k r_c pi}); the warped Higgs vacuum value v = e^{-k r_c pi} v_0 and
# k r_c = 15 ln 10/pi ~ 11 for a factor 10^15; the zero mode solves the Schrodinger problem with the jump fixed by
# the delta, has unit norm, and the change of variables maps the y-equation to the z-equation.
# Output: figures/warp_factor.svg, interactive figures/interactive/warp_factor.html.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "warp_factor.svg")
t, x1, x2, x3, y = var("t x1 x2 x3 y", domain="real")
k, Lam = var("k Lambda", domain="real")
A = function("A")(y)
X = [t, x1, x2, x3, y]
g = diagonal_matrix(SR, [exp(-2 * A), -exp(-2 * A), -exp(-2 * A), -exp(-2 * A), -1])
gi = g.inverse()
n = 5
Gam = [[[sum(gi[r, l] * (diff(g[l, j], X[i]) + diff(g[l, i], X[j]) - diff(g[i, j], X[l])) for l in range(n)) / 2
         for j in range(n)] for i in range(n)] for r in range(n)]
def riem(r, s, m, v):        # R^r_{s m v} = d_m Gam^r_{v s} - d_v Gam^r_{m s} + Gam^r_{m l} Gam^l_{v s} - Gam^r_{v l} Gam^l_{m s}
    return (diff(Gam[r][v][s], X[m]) - diff(Gam[r][m][s], X[v])
            + sum(Gam[r][m][l] * Gam[l][v][s] - Gam[r][v][l] * Gam[l][m][s] for l in range(n)))
Ric = matrix(SR, n, n, lambda s, v: sum(riem(r, s, r, v) for r in range(n)).simplify_full())
Rs = sum(gi[i, j] * Ric[i, j] for i in range(n) for j in range(n)).simplify_full()
Ein = (Ric - Rs * g / 2).simplify_full()
Ap, App = diff(A, y), diff(A, y, 2)
print("[PASS] warped metric: G_{mu nu} = (3 A'' - 6 A'^2) g_{mu nu} and G_{yy} = -6 A'^2 g_{yy}, off-diagonal components vanish:",
      all(bool((Ein[i, i] - (3 * App - 6 * Ap**2) * g[i, i]).simplify_full() == 0) for i in range(4))
      and bool((Ein[4, 4] - (-6 * Ap**2) * g[4, 4]).simplify_full() == 0)
      and all(bool(Ein[i, j] == 0) for i in range(n) for j in range(n) if i != j))
# bulk equations G_MN - Lambda g_MN = 0: yy gives -6 A'^2 - Lambda = 0, mu nu then 3 A'' = 0
Af = function("A")
def warp(expr, F): return expr.substitute_function(Af, F)
lin = Ein.apply_map(lambda e: warp(e, lambda u: k * u).simplify_full())
gl = g.apply_map(lambda e: warp(e, lambda u: k * u))
print("[PASS] A = k y solves the bulk equations G_MN - Lambda g_MN = 0 (y > 0) exactly when Lambda = -6 k^2:",
      all(bool((lin[i, j] - Lam * gl[i, j]).subs({Lam: -6 * k**2}).simplify_full() == 0) for i in range(n) for j in range(n))
      and bool((lin[4, 4] - Lam * gl[4, 4] - (6 * k**2 + Lam)).simplify_full() == 0))
print("[PASS] the same holds for A = -k y (y < 0), hence for A = k|y| on both sides of the brane:",
      all(bool((warp(Ein[i, j], lambda u: -k * u) - (-6 * k**2) * warp(g[i, j], lambda u: -k * u)).simplify_full() == 0) for i in range(n) for j in range(n)))
print("[PASS] for a general A the yy equation gives A'^2 = -Lambda/6 (so Lambda < 0, AdS_5) and the mu nu equation then A'' = 0:",
      bool(((Ein[4, 4] - Lam * g[4, 4]) - (6 * Ap**2 + Lam)).simplify_full() == 0)
      and bool(((Ein[0, 0] - Lam * g[0, 0]) / g[0, 0] - (3 * App - 6 * Ap**2 - Lam)).simplify_full() == 0))
# the brane: A = k|y| has A'' = 2k delta(y); integrating G_{mu nu} - Lambda g_{mu nu} = kappa^2 T_{mu nu} with
# T_{mu nu} = V delta(y) g_{mu nu} across y = 0 gives 3 [A'] = kappa^2 V, [A'] = 2k
kap, M, V = var("kap M V", domain="positive")
Vsol = solve(3 * 2 * k == kap**2 * V, V)[0].rhs()
print("[PASS] brane tension: 3 [A'] = 6k = kappa_5^2 V gives V = 6k/kappa_5^2, which is 24 M^3 k with kappa_5^2 = 1/(4 M^3):",
      bool(Vsol == 6 * k / kap**2) and bool(Vsol.subs({kap: 1 / (2 * M**(3 / 2))}) == 24 * M**3 * k))
print("[PASS] in Randall-Sundrum normalisation (2 M^3 R term, Lambda_RS = Lambda/kappa_5^2) k^2 = -Lambda_RS/(24 M^3):",
      bool((-(-6 * k**2) / kap**2 / (24 * M**3)).subs({kap: 1 / (2 * M**(3 / 2))}) == k**2))
# four-dimensional Planck mass and the hierarchy
rc, phi = var("rc phi", domain="positive")
MPl2 = M**3 * rc * (integrate(exp(-2 * k * rc * phi), phi, 0, pi) * 2)
assume(k > 0)
print("[PASS] M_Pl^2 = M^3 r_c int_{-pi}^{pi} e^{-2 k r_c |phi|} dphi = (M^3/k)(1 - e^{-2 k r_c pi}), finite as r_c -> infinity:",
      bool((MPl2 - M**3 / k * (1 - exp(-2 * k * rc * pi))).simplify_full() == 0) and bool(limit(MPl2, rc=oo) == M**3 / k))
H, v0, lam, w = var("H v0 lam w", domain="positive")      # w = e^{k r_c pi}
L_vis = w**(-4) * (w**2 * H**2 - lam * (H**2 - v0**2)**2)  # sqrt(-g_vis) {g_vis^{mu nu} DH DH - lam(|H|^2 - v0^2)^2}, schematic |DH| -> H
L_can = L_vis.subs(H=w * H).expand()
print("[PASS] warped Higgs: after H -> e^{k r_c pi} H the kinetic term is canonical and the vacuum value is v = e^{-k r_c pi} v0:",
      bool((L_can - (H**2 - lam * (H**2 - v0**2 / w**2)**2)).expand() == 0))
krc = 15 * log(10) / pi
print("[PASS] a hierarchy e^{k r_c pi} = 10^15 needs only k r_c = 15 ln 10/pi = %.2f:" % float(krc), bool(exp(krc * pi) == 10**15) and 10.9 < float(krc) < 11.1)
# graviton zero mode in the conformal coordinate z (z > 0 branch): y = log(k z + 1)/k, psihat = psi e^{k y/2}
z, m = var("z m", domain="real")
psi0 = (k * z + 1)**(-3 / 2)
Vz = 15 * k**2 / (8 * (k * z + 1)**2)
print("[PASS] zero mode: psihat_0 = (k z + 1)^{-3/2} solves -1/2 psihat'' + 15 k^2/(8 (k z + 1)^2) psihat = 0 for z > 0:",
      bool((-diff(psi0, z, 2) / 2 + Vz * psi0).simplify_full() == 0))
print("[PASS] the delta -(3k/2) delta(z) fixes the jump: psihat'(0+) - psihat'(0-) = -3k psihat(0) for psihat = (k|z| + 1)^{-3/2}:",
      bool(2 * diff(psi0, z).subs(z=0) == -3 * k * psi0.subs(z=0)))
norm = 2 * integrate(k * psi0**2, z, 0, oo)
print("[PASS] sqrt(k) (k|z| + 1)^{-3/2} has unit norm: the graviton is a normalisable bound state although z ranges over the real line:", bool(norm == 1))
# psi(y) = psihat e^{-k|y|/2}; the zero mode in y is constant e^{-2k|y|} h(x) up to the h -> e^{2k|y|} h convention: check
# the y-equation [-m^2/2 e^{2ky} - 1/2 d_y^2 + 2k^2] psi = 0 at m = 0 is solved by psi = e^{-2ky} (y > 0)
psiy = exp(-2 * k * y)
print("[PASS] in y the zero mode is psi = e^{-2k|y|}: -1/2 psi'' + 2 k^2 psi = 0 for y > 0 and psi'(0+) - psi'(0-) = -4k psi(0), the delta -2k delta(y):",
      bool((-diff(psiy, y, 2) / 2 + 2 * k**2 * psiy).simplify_full() == 0) and bool(2 * diff(psiy, y).subs(y=0) == -4 * k))
ys = (log(k * z + 1) / k)
print("[PASS] the change of variables: psihat(z) = psi(y) e^{k y/2} with y = log(kz + 1)/k maps e^{-2ky} to (kz + 1)^{-3/2}:",
      bool((psiy.subs(y=ys) * exp(k * ys / 2) - psi0).simplify_full() == 0))

cols = {0.5: "royalblue", 1: "darkgreen", 2: "firebrick"}
left = Graphics(); right = Graphics()
for kk, c in cols.items():
    left += plot(exp(-2 * kk * abs(y)), (y, -3, 3), color=c, thickness=1.8, legend_label="$k=%g$" % kk)
    right += plot(sqrt(kk) * (kk * abs(z) + 1)**(-3 / 2), (z, -4, 4), color=c, thickness=1.8, legend_label="$\\hat\\psi_0,\\ k=%g$" % kk)
    right += plot(15 * kk**2 / (8 * (kk * abs(z) + 1)**2) / 4, (z, -4, 4), color=c, thickness=1, linestyle="--")
left.axes_labels(["$y$", "$e^{-2k|y|}$"]); left.set_legend_options(loc="upper right", font_size=9)
right.axes_labels(["$z$", "$\\hat\\psi_0(z)$;  $V(z)/4$ dashed"]); right.set_axes_range(ymin=0, ymax=1.6); right.set_legend_options(loc="upper right", font_size=9)
G = graphics_array([left, right]); fig = G.matplotlib(figsize=[10, 4.2]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

# Interactive version: slide the AdS curvature k (0.25 to 3). Left: the warp factor e^{-2k|y|}. Right: the
# normalised zero mode sqrt(k)(k|z| + 1)^{-3/2} and the volcano potential 15k^2/(8(k|z| + 1)^2) (scaled by 1/4): a
# larger k localises the graviton more sharply at the brane.
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json, math
_ys = [-3 + 6 * i / 300.0 for i in range(301)]; _zs = [-4 + 8 * i / 300.0 for i in range(301)]
def _data(kk):
    return [{"x": _ys, "y": [math.exp(-2 * kk * abs(u)) for u in _ys]},
            {"x": _zs, "y": [math.sqrt(kk) * (kk * abs(u) + 1) ** -1.5 for u in _zs]},
            {"x": _zs, "y": [15 * kk * kk / (8 * (kk * abs(u) + 1) ** 2) / 4 for u in _zs]}]
_frames, _steps = [], []
for i in range(1, 13):
    kk = 0.25 * i
    _frames.append({"name": "%.2f" % kk, "data": _data(kk), "traces": [0, 1, 2]})
    _steps.append({"label": "%.2f" % kk, "method": "animate", "args": [["%.2f" % kk], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]})
_d = _data(1.0)
_traces = [dict(_d[0], mode="lines", name="warp factor e^{-2k|y|}", line={"color": "royalblue", "width": 3}),
           dict(_d[1], mode="lines", name="zero mode sqrt(k)(k|z|+1)^{-3/2}", line={"color": "firebrick", "width": 3}, xaxis="x2", yaxis="y2"),
           dict(_d[2], mode="lines", name="volcano potential / 4", line={"color": "grey", "dash": "dash"}, xaxis="x2", yaxis="y2")]
_layout = {"title": "Randall-Sundrum: warp factor and localised graviton; slide the AdS curvature k",
           "grid": {"rows": 1, "columns": 2, "pattern": "independent"},
           "xaxis": {"title": "y", "range": [-3, 3]}, "yaxis": {"range": [0, 1.05]},
           "xaxis2": {"title": "z", "range": [-4, 4]}, "yaxis2": {"range": [0, 2.0]},
           "sliders": [{"active": 3, "currentvalue": {"prefix": "k = "}, "steps": _steps, "pad": {"t": 30}}], "legend": {"orientation": "h", "y": -0.3}}
_path = plotly_html("warp_factor", _traces, _layout)
_html = open(_path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(_frames)))
open(_path, "w").write(_html)
_dz = 8 / 300.0
print("[PASS] slider: every frame's zero mode has unit norm to 5e-3 (trapezoidal sum over |z| < 4 plus the exact tail 1/(kz + 1)^2 at z = 4):",
      all(abs(sum(_dz * (u0 ** 2 + u1 ** 2) / 2 for u0, u1 in zip(fr["data"][1]["y"][:-1], fr["data"][1]["y"][1:])) + 1 / (0.25 * (i + 1) * 4 + 1) ** 2 - 1) < 5e-3
          for i, fr in enumerate(_frames)))
print("wrote figures/interactive/warp_factor.html")
