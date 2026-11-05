# Figure: the radii of Freund-Rubin vacua. Left: for AdS_4 x S^7 with flux F = 3m eps, the anti-de Sitter radius
# L = 1/(2m) and the sphere radius a = 1/m against m; the sphere is always twice the AdS radius. Right: the ratio
# a/L = (q-1)/(p-1) of AdS_p x S^q supported by a p-form flux, for the maximally supersymmetric vacua AdS_4 x S^7,
# AdS_5 x S^5, AdS_7 x S^4 and for AdS_2 x S^2, AdS_3 x S^3. The checks compute the Ricci tensors of the explicit
# metrics (Poincare AdS_4 and the round S^7 in hyperspherical angles) in the signature (+, -, ..., -) and verify the
# eleven-dimensional Einstein equations; they also find the integer solutions of the sphere-consistency relations of
# Cvetic-Lu-Pope and Hohm-Samtleben. Output: figures/vcr_freund_rubin.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vcr_freund_rubin.svg")
mm = var("mm")
left = plot(1 / mm, (mm, 0.25, 2), color="royalblue", thickness=2, legend_label=r"$S^7$ radius $a = 1/m$")
left += plot(1 / (2 * mm), (mm, 0.25, 2), color="firebrick", thickness=2, legend_label=r"$AdS_4$ radius $L = 1/2m$")
left.axes_labels([r"$m$", "radius"]); left.set_legend_options(loc="upper right", font_size=8)
cases = [("$AdS_2\\times S^2$", 2, 2), ("$AdS_3\\times S^3$", 3, 3), ("$AdS_4\\times S^7$", 4, 7), ("$AdS_5\\times S^5$", 5, 5), ("$AdS_7\\times S^4$", 7, 4)]
right = Graphics()
for i, (name, p, q) in enumerate(cases):
    r = QQ(q - 1) / (p - 1)
    right += polygon([(i - 0.3, 0), (i + 0.3, 0), (i + 0.3, r), (i - 0.3, r)], color="darkgreen" if (p, q) in [(4, 7), (5, 5), (7, 4)] else "gray", alpha=0.7)
    right += text(name, (i, -0.15), fontsize=8)
    right += text("$%s$" % latex(r), (i, r + 0.12), fontsize=9)
right.axes_labels(["", r"$a/L$"]); right.ymin(-0.3); right.ymax(2.4); right._extra_kwds.update(ticks=[[], None])
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[10, 4], frame=True, axes=False); fig.savefig(OUT, bbox_inches="tight")

m = var("m", domain="positive")
L, a = 1 / (2 * m), 1 / m
# AdS_4 in Poincare coordinates, signature (+, -, -, -): ds^2 = (L^2/z^2)(dt^2 - dx^2 - dy^2 - dz^2)
A4 = Manifold(4, "AdS4", structure="Lorentzian", signature=-2); XA = A4.chart("t x y z:(0,+oo)"); t, x, yv, z = XA[:]
gA = A4.metric("gA"); gA[0, 0] = L ** 2 / z ** 2
for i in (1, 2, 3):
    gA[i, i] = -L ** 2 / z ** 2
RicA = gA.ricci()
okA = all(bool((RicA[i, j].expr() - 12 * m ** 2 * gA[i, j].expr()).simplify_full() == 0) for i in range(4) for j in range(4))
# the round S^7 of radius a in hyperspherical angles; internal metric h = -a^2 (round unit metric)
S7 = Manifold(7, "S7"); XS = S7.chart(" ".join("c%d" % k for k in range(7))); c = XS[:]
h = S7.metric("h", signature=-7)
w = 1
for k in range(7):
    h[k, k] = -a ** 2 * w
    w = w * sin(c[k]) ** 2
RicS = h.ricci()
okS = all(bool((RicS[i, j].expr() + 6 * m ** 2 * h[i, j].expr()).simplify_full() == 0) for i in range(7) for j in range(i, 7))
print("[PASS] explicit metrics in signature (+,-,...,-): Poincare AdS_4 of radius L = 1/(2m) has R_mu nu = 12 m^2 g_mu nu and the round S^7 of radius a = 1/m has R_mn = -6 m^2 h_mn:", okA and okS)
# the four-form F = 3m eps on AdS_4 and the eleven-dimensional Einstein equations
G4 = matrix(SR, 4, 4, lambda i, j: gA[i, j].expr()); G4i = G4.inverse()
sq = sqrt(-G4.det()).simplify_full()
def Feps(idx):
    return 3 * m * sq * (Permutation([k + 1 for k in idx]).sign() if len(set(idx)) == 4 else 0)
from itertools import permutations
perms = list(permutations(range(4)))
FF = matrix(SR, 4, 4, lambda mu, nu: sum(Feps((mu,) + p[1:]) * Feps((nu,) + tuple(q[1:])) * prod(G4i[p[k], q[k]] for k in (1, 2, 3))
                                           for p in perms if p[0] == mu for q in perms if q[0] == nu) / 1)
# the sum over ordered index triples counts each (alpha, beta, gamma) once since p, q run over permutations with fixed first entry
F2 = sum(FF[mu, nu] * G4i[mu, nu] for mu in range(4) for nu in range(4)) * 1
FF = FF.simplify_full(); F2 = F2.simplify_full()
okF = all(bool((FF[i, j] + 54 * m ** 2 * G4[i, j]).simplify_full() == 0) for i in range(4) for j in range(4)) and bool(F2 == -216 * m ** 2)
Rtot = (sum(RicA[i, j].expr() * G4i[i, j] for i in range(4) for j in range(4)) + sum(RicS[i, i].expr() / h[i, i].expr() for i in range(7))).simplify_full()
einA = all(bool((RicA[i, j].expr() - G4[i, j] * Rtot / 2 + (FF[i, j] - G4[i, j] * F2 / 8) / 3).simplify_full() == 0) for i in range(4) for j in range(4))
einS = all(bool((RicS[i, i].expr() - h[i, i].expr() * Rtot / 2 + (0 - h[i, i].expr() * F2 / 8) / 3).simplify_full() == 0) for i in range(7))
print("[PASS] with F_{mu nu rho sigma} = 3m eps: F_{mu PQR} F_nu^{PQR} = -54 m^2 g_mu nu, F^2 = -216 m^2, R = 6 m^2, and R_MN - g_MN R/2 = -(1/3)(F_MPQR F_N^PQR - g_MN F^2/8) holds on AdS_4 x S^7 component by component:",
      okF and bool(Rtot == 6 * m ** 2) and einA and einS)
dF = all(bool(diff(sq * prod(G4i[k, k] for k in range(4)) * Feps((0, 1, 2, 3)), v) == 0) for v in (t, x, yv))
print("[PASS] the flux equation d * F = 0: sqrt|g| F^{0123} = -3m is constant on AdS_4, so its divergence vanishes (F ^ F = 0 since F lives on four dimensions):",
      dF and bool(diff(sq * prod(G4i[k, k] for k in range(4)) * Feps((0, 1, 2, 3)), z) == 0))
pp, qq, cc = var("pp qq cc", domain="positive")
Dd = pp + qq
fact = factorial(pp - 1)                                    # (p-1)!, cancels below
FFmu = -fact * cc ** 2; F2c = -pp * fact * cc ** 2              # mostly-plus: eps_mu... eps_nu^... = -(p-1)! g, F^2 = -p! c^2
Rmunu = ((FFmu - (pp - 1) / (pp * (Dd - 2)) * F2c) / (2 * fact)).simplify_full()   # trace-reversed Einstein equation
Rmn = ((0 - (pp - 1) / (pp * (Dd - 2)) * F2c) / (2 * fact)).simplify_full()
ratio2 = (((qq - 1) / Rmn) / ((pp - 1) / (-Rmunu))).simplify_full()       # a^2 / L^2
print("[PASS] AdS_p x S^q with a p-form flux c eps_p: R_mu nu = -(q-1)c^2/(2(D-2)) g, R_mn = (p-1)c^2/(2(D-2)) g, hence a/L = (q-1)/(p-1): 2, 1, 1/2 for (p, q) = (4, 7), (5, 5), (7, 4):",
      bool((Rmunu + cc ** 2 * (qq - 1) / (2 * (Dd - 2))).simplify_full() == 0) and bool((Rmn - cc ** 2 * (pp - 1) / (2 * (Dd - 2))).simplify_full() == 0)
      and bool((ratio2 - ((qq - 1) / (pp - 1)) ** 2).simplify_full() == 0) and [sqrt(ratio2.subs(pp=P, qq=Q)) for P, Q in [(4, 7), (5, 5), (7, 4)]] == [2, 1, 1 / 2])
sols = [(D, p) for p in range(2, 40) for D in range(p + 2, 60) if 2 * D - 4 == (p - 1) * (D - p - 1)]
print("[PASS] the relation 2D - 4 = (p-1)(D-p-1) of Cvetic-Lu-Pope has exactly the solutions (D, p) = (11, 4), (10, 5), (11, 7) for 2 <= p, D >= p + 2 (search up to D = 59; D = p + 3 + 4/(p - 3) in general):",
      sorted(sols) == [(10, 5), (11, 4), (11, 7)])
DD, P = var("DD P")
a2 = 4 - 2 * (P - 1) * (DD - P - 1) / (DD - 2)
print("[PASS] the dilaton coupling a^2 = 4 - 2(p-1)(D-p-1)/(D-2) equals 2(D-1)/(D-2) for p = 2 and 8/(D-2) for p = 3, positive in every D:",
      bool((a2.subs(P=2) - 2 * (DD - 1) / (DD - 2)).simplify_full() == 0) and bool((a2.subs(P=3) - 8 / (DD - 2)).simplify_full() == 0))
hs = [(D, n) for n in range(5, 200) for D in range(2, 30) if QQ(D - 1) / 2 == QQ(n - 2) / (n - 4)]
print("[PASS] the relation (D-1)/2 = (n-2)/(n-4) of Hohm-Samtleben has exactly the solutions (D, n) = (7, 5), (5, 6), (4, 8) for n > 4:", hs == [(7, 5), (5, 6), (4, 8)])
print("wrote", os.path.relpath(OUT))

# Interactive version: cross-sections of the two factors drawn to scale, a circle of radius a = 1/m for S^7 and one of
# radius L = 1/(2m) for AdS_4, with a slider for m.
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json, math
th = [2 * math.pi * k / 120 for k in range(121)]
def _data(mv):
    A, Lr = 1 / mv, 1 / (2 * mv)
    return [{"x": [A * math.cos(s) - 2.2 for s in th], "y": [A * math.sin(s) for s in th]},
            {"x": [Lr * math.cos(s) + 2.2 for s in th], "y": [Lr * math.sin(s) for s in th]}]
_vals = [round(0.5 + 0.1 * j, 1) for j in range(0, 16)]
_frames, _steps = [], []
for mv in _vals:
    _frames.append({"name": "%.1f" % mv, "data": _data(mv), "traces": [0, 1]})
    _steps.append({"label": "%.1f" % mv, "method": "animate", "args": [["%.1f" % mv], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]})
_d0 = _data(1.0)
_traces = [dict(_d0[0], mode="lines", name="S^7 radius a = 1/m", line={"color": "royalblue", "width": 3}),
           dict(_d0[1], mode="lines", name="AdS_4 radius L = 1/(2m)", line={"color": "firebrick", "width": 3})]
_layout = {"title": "Freund-Rubin AdS_4 x S^7: the sphere is twice the AdS radius for every flux m",
           "xaxis": {"range": [-4.6, 4.6], "zeroline": False}, "yaxis": {"range": [-2.2, 2.2], "scaleanchor": "x", "zeroline": False},
           "sliders": [{"active": 5, "currentvalue": {"prefix": "m = "}, "steps": _steps, "pad": {"t": 40}}], "legend": {"orientation": "h", "y": -0.3}}
_path = plotly_html("vcr_freund_rubin", _traces, _layout)
open(_path, "w").write(open(_path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(_frames))))
print("wrote figures/interactive/vcr_freund_rubin.html")
