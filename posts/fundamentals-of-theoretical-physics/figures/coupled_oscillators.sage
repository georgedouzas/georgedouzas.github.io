# Figure: normal modes of two equal masses m between two walls, each tied to its wall by a spring k and to the other
# mass by a spring kappa (units m = k = 1, kappa = 0.2). Left: level curves of the potential V = x^T K x/2 in the
# displacement plane (x1, x2), the two normal-mode directions (1, 1) and (1, -1), and the trajectory that starts at
# x = (1, 0) at rest. Right: the same motion against time, a beat in which the energy passes from one mass to the
# other with period 2 pi/(omega2 - omega1). Further checks of the section on small oscillations: the triatomic
# molecule, separate conservation of the mode energies, the spherical pendulum and the particle on SU(2) = S^3.
# Output: figures/coupled_oscillators.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coupled_oscillators.svg")
m, k, kap, t = var("m k kappa t", domain="positive")
M = m * identity_matrix(2)
K = matrix([[k + kap, -kap], [-kap, k + kap]])
D = M.inverse() * K
ev = D.eigenvectors_right()
evals = sorted([e_[0] for e_ in ev], key=lambda s: s.subs(m=1, k=1, kappa=1))
okvec = all(bool(((D - e_[0]) * e_[1][0]).simplify_full() == 0) for e_ in ev)
print("[PASS] two masses m with springs k, kappa, k: omega^2 = k/m and (k + 2 kappa)/m, with modes (1, 1) and (1, -1):", bool(evals[0] == k / m) and bool((evals[1] - (k + 2 * kap) / m).simplify_full() == 0) and okvec and bool(K * vector([1, 1]) == (k / m) * M * vector([1, 1])) and bool(K * vector([1, -1]) == ((k + 2 * kap) / m) * M * vector([1, -1])))
w1, w2 = sqrt(k / m), sqrt((k + 2 * kap) / m)
X = vector([(cos(w1 * t) + cos(w2 * t)) / 2, (cos(w1 * t) - cos(w2 * t)) / 2])
res = M * X.apply_map(lambda f: diff(f, t, 2)) + K * X
print("[PASS] the beat x = ((cos w1 t + cos w2 t)/2, (cos w1 t - cos w2 t)/2) solves M x'' + K x = 0 with x(0) = (1, 0), x'(0) = 0:", all(bool(r_.simplify_full() == 0) for r_ in res) and bool(X.subs(t=0) == vector([1, 0])) and bool(X.apply_map(lambda f: diff(f, t)).subs(t=0) == vector([0, 0])))
# mode energies: with Q1 = (x1 + x2)/sqrt 2, Q2 = (x1 - x2)/sqrt 2, E = sum (m Q'^2 + m w^2 Q^2)/2, each conserved
Q1 = (X[0] + X[1]) / sqrt(2); Q2 = (X[0] - X[1]) / sqrt(2)
E1 = m * (diff(Q1, t)**2 + w1**2 * Q1**2) / 2; E2 = m * (diff(Q2, t)**2 + w2**2 * Q2**2) / 2
Etot = m * (diff(X[0], t)**2 + diff(X[1], t)**2) / 2 + (X * K * X) / 2
print("[PASS] the energies of the two normal modes are separately conserved and add up to the total energy:", bool(diff(E1, t).simplify_full() == 0) and bool(diff(E2, t).simplify_full() == 0) and bool((Etot - E1 - E2).simplify_full() == 0))
# linear triatomic molecule m - M - m with springs k: frequencies 0, k/m, k(1/m + 2/M)
mM = var("mM", domain="positive")
M3 = diagonal_matrix([m, mM, m]); K3 = matrix([[k, -k, 0], [-k, 2 * k, -k], [0, -k, k]])
lam = var("lam")
ch = (M3.inverse() * K3 - lam * identity_matrix(3)).det()
print("[PASS] the linear triatomic molecule (masses m, M, m, springs k): omega^2 = 0, k/m, k(1/m + 2/M):", bool((-ch - lam * (lam - k / m) * (lam - k * (1 / m + 2 / mM))).simplify_full() == 0))
# spherical pendulum of length l: chart (x, y) near the bottom, z = -sqrt(l^2 - x^2 - y^2), V = m g z
g, l = var("g l", domain="positive")
x, y = var("x y", domain="real")
Zs = -sqrt(l**2 - x**2 - y**2)
Emb = vector([x, y, Zs]); Jm = jacobian(Emb, [x, y]); G = (Jm.transpose() * Jm).subs(x=0, y=0)
Hs = jacobian(jacobian(m * g * Zs, [x, y]), [x, y]).subs(x=0, y=0)
print("[PASS] spherical pendulum: at the bottom of the sphere g^{-1} V'' = (g/l) times the identity, two modes of frequency sqrt(g/l):", bool((m * G).inverse() * Hs == (g / l) * identity_matrix(2)))
# particle on SU(2) = S^3 (unit radius, the metric of the Euler-angle Gram matrix) with V = kk (1 - Re tr U/2) = kk (1 - a0)
a1, a2, a3, kk = var("a1 a2 a3 kk", domain="real")
a0 = sqrt(1 - a1**2 - a2**2 - a3**2)
E4 = vector([a0, a1, a2, a3]); J4 = jacobian(E4, [a1, a2, a3]); G4 = (J4.transpose() * J4).subs(a1=0, a2=0, a3=0)
H4 = jacobian(jacobian(kk * (1 - a0), [a1, a2, a3]), [a1, a2, a3]).subs(a1=0, a2=0, a3=0)
print("[PASS] particle on SU(2) with V = k(1 - Re tr U/2): at U = 1, g^{-1} V'' = k times the identity, three degenerate modes:", bool(G4.inverse() * H4 == kk * identity_matrix(3)))
b0 = var("b0", domain="real")
U2 = matrix(SR, [[b0 + I * a3, a2 + I * a1], [-a2 + I * a1, b0 - I * a3]])
nrm = b0**2 + a1**2 + a2**2 + a3**2
print("[PASS] U = a0 + i a.sigma has U^dagger U = det U = a0^2 + |a|^2 (so U is in SU(2) on the unit S^3) and Re tr U/2 = a0:", bool((U2.conjugate_transpose() * U2).expand() == nrm * identity_matrix(2)) and bool(U2.det().expand() == nrm) and bool((U2.trace() / 2).real() == b0))

import numpy as np
k_, kap_ = 1.0, 0.2
W1, W2 = 1.0, float(np.sqrt(1.4))
ts = np.linspace(0, 70, 1400)
x1 = (np.cos(W1 * ts) + np.cos(W2 * ts)) / 2; x2 = (np.cos(W1 * ts) - np.cos(W2 * ts)) / 2
u, v = var("u v")
left = contour_plot(((k_ + kap_) * (u**2 + v**2) - 2 * kap_ * u * v) / 2, (u, -1.2, 1.2), (v, -1.2, 1.2), contours=[0.05, 0.2, 0.45, 0.8], fill=False, cmap=["lightgrey"])
left += line(list(zip(x1[:700], x2[:700])), color="royalblue", thickness=0.8, alpha=0.8)
left += arrow((0, 0), (0.9, 0.9), color="darkgreen", width=1.5) + arrow((0, 0), (0.9, -0.9), color="firebrick", width=1.5)
left += text(r"mode 1: $(1,1)$, $\omega_1^2=k/m$", (-0.15, 1.1), color="darkgreen", fontsize=9)
left += text(r"mode 2: $(1,-1)$, $\omega_2^2=(k+2\kappa)/m$", (-0.1, -1.12), color="firebrick", fontsize=9)
left.axes_labels([r"$x_1$", r"$x_2$"]); left.set_aspect_ratio(1)
right = line(list(zip(ts, x1)), color="royalblue", legend_label=r"$x_1(t)$") + line(list(zip(ts, x2 - 2.5)), color="darkorange", legend_label=r"$x_2(t)$ (shifted by $-2.5$)")
right.axes_labels([r"$t$", ""]); right.set_legend_options(loc="upper right", font_size=8)
gfx = graphics_array([left, right]); fig = gfx.matplotlib(figsize=[11, 4.3]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

# interactive animation: the two masses and three springs, with buttons for mode 1, mode 2 and the beat
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json
def spring(xa, xb, n=10, amp=0.08):
    xs = list(np.linspace(xa, xb, 2 * n + 1)); ys = [0.0] + [amp * (-1)**j for j in range(1, 2 * n)] + [0.0]
    return xs, ys
def state(d1, d2):
    p1, p2 = 1 + 0.35 * d1, 2 + 0.35 * d2
    out = []
    for xa, xb in [(0, p1), (p1, p2), (p2, 3)]:
        xs, ys = spring(xa, xb); out.append({"x": xs, "y": ys})
    out.append({"x": [p1, p2], "y": [0, 0]})
    return out
frames, groups = [], {}
for key, (T, nf, f) in {"mode1": (2 * np.pi / W1, 40, lambda s: (np.cos(W1 * s), np.cos(W1 * s))),
                        "mode2": (2 * np.pi / W2, 40, lambda s: (np.cos(W2 * s), -np.cos(W2 * s))),
                        "beat": (2 * np.pi / (W2 - W1), 300, lambda s: ((np.cos(W1 * s) + np.cos(W2 * s)) / 2, (np.cos(W1 * s) - np.cos(W2 * s)) / 2))}.items():
    names = []
    for j in range(nf + 1):
        s = T * j / nf; d = f(s); nm = "%s-%d" % (key, j)
        frames.append({"name": nm, "data": state(*d), "traces": [0, 1, 2, 3]}); names.append(nm)
    groups[key] = names
init = state(1, 1)
traces = [{"x": init[i]["x"], "y": init[i]["y"], "mode": "lines", "line": {"color": c, "width": 2}, "name": n_, "hoverinfo": "skip"} for i, (c, n_) in enumerate([("grey", "spring k"), ("purple", "coupling spring kappa"), ("grey", "spring k ")])]
traces.append({"x": init[3]["x"], "y": init[3]["y"], "mode": "markers", "marker": {"size": 26, "color": ["royalblue", "darkorange"]}, "name": "masses"})
traces.append({"x": [0, 0, None, 3, 3], "y": [-0.3, 0.3, None, -0.3, 0.3], "mode": "lines", "line": {"color": "black", "width": 6}, "name": "walls", "hoverinfo": "skip"})
def button(label, key, dur):
    return {"label": label, "method": "animate", "args": [groups[key], {"mode": "immediate", "fromcurrent": False, "frame": {"duration": dur, "redraw": False}, "transition": {"duration": 0}}]}
layout = {"title": "Two coupled oscillators (k = m = 1, kappa = 0.2): press a button",
          "xaxis": {"range": [-0.2, 3.2], "title": "position", "showgrid": False}, "yaxis": {"range": [-0.6, 0.6], "visible": False, "scaleanchor": "x"},
          "updatemenus": [{"type": "buttons", "direction": "left", "x": 0.0, "y": -0.25, "xanchor": "left", "buttons": [
              button("mode 1 (in phase)", "mode1", 60), button("mode 2 (out of phase)", "mode2", 60), button("beat", "beat", 40)]}],
          "legend": {"orientation": "h"}}
path = plotly_html("coupled_oscillators", traces, layout)
html = open(path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(frames)))
open(path, "w").write(html)
print("wrote figures/interactive/coupled_oscillators.html")
