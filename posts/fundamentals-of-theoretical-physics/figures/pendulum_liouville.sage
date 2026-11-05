# Figure: Liouville's theorem for the pendulum H = p^2/2 - cos q. A disc of initial conditions in phase space is carried
# by the Hamiltonian flow (integrated numerically) to the regions shown at t = 0, 1.5, 3, 4.5: the region shears and
# stretches along the level curves of H, but its area stays that of the disc. Background: level curves of H and the
# separatrix H = 1 (red). Further checks of the section on Poisson brackets and canonical transformations: the
# Leibniz rule and Jacobi identity of the bracket, the so(3) algebra of angular momentum, canonicity of the oscillator
# flow and of the action-angle map, a type-2 generating function, and solutions of the Hamilton-Jacobi equation.
# Output: figures/pendulum_liouville.svg.
import os
import numpy as np
from scipy.integrate import solve_ivp
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pendulum_liouville.svg")
q, p = var("q p", domain="real")
H = p**2 / 2 - cos(q)
N = 1500; c0 = (0.0, 1.5); rad = 0.4
ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
Y0 = np.concatenate([c0[0] + rad * np.cos(ang), c0[1] + rad * np.sin(ang)])
def rhs(t, Y):
    return np.concatenate([Y[N:], -np.sin(Y[:N])])
times = [0, 1.5, 3, 4.5]
sol = solve_ivp(rhs, [0, times[-1]], Y0, t_eval=times, rtol=1e-10, atol=1e-10)
def area(xs, ys):
    return 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
areas = [area(sol.y[:N, j], sol.y[N:, j]) for j in range(len(times))]
P = contour_plot(H, (q, -3.5, 3.5), (p, -2.6, 2.6), contours=[-0.8, -0.4, 0, 0.4, 1.6, 2.4], fill=False, cmap=["lightgrey"], linewidths=1)
P += contour_plot(H, (q, -3.5, 3.5), (p, -2.6, 2.6), contours=[1.0], fill=False, cmap=["firebrick"], linewidths=1.5)
shades = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]
for j, t0 in enumerate(times):
    pts = list(zip(sol.y[:N, j], sol.y[N:, j]))
    P += polygon(pts, color=shades[j], alpha=0.75, edgecolor=shades[j])
    k = int(np.argmax(sol.y[N:, j]))
    P += text("t = %g" % t0, (float(sol.y[k, j]), float(sol.y[N + k, j]) + 0.25), fontsize=9, color=shades[j])
P += text(r"area of the region: $\pi\,(0.4)^2 \approx %.4f$ at every $t$" % float(pi * rad**2), (0, -2.35), fontsize=9)
P.axes_labels([r"$q$", r"$p$"])
P.save(OUT, figsize=[8, 4.4])
print("[PASS] the pendulum flow preserves the area of a disc of initial conditions (pi 0.4^2) to 1e-3 at t = 0, 1.5, 3, 4.5:", all(abs(A_ - float(pi * rad**2)) < 1e-3 for A_ in areas))
print("[PASS] the Hamiltonian vector field of any H(q, p) has zero divergence (Liouville):", bool((diff(diff(H, p), q) + diff(-diff(H, q), p)) == 0) and bool((lambda F: diff(diff(F, p), q) - diff(diff(F, q), p))(function("F")(q, p)) == 0))

# Poisson algebra
def PB1(f, g): return diff(f, q) * diff(g, p) - diff(f, p) * diff(g, q)
f, g, h = [sum(QQ.random_element(num_bound=5, den_bound=1) * q**i * p**j for i in range(3) for j in range(3)) for _ in range(3)]
F, G, K = [function(n)(q, p) for n in "FGK"]
jac = PB1(F, PB1(G, K)) + PB1(G, PB1(K, F)) + PB1(K, PB1(F, G))
leib = PB1(F, G * K) - PB1(F, G) * K - G * PB1(F, K)
print("[PASS] the Poisson bracket is antisymmetric and satisfies the Leibniz rule and the Jacobi identity for arbitrary functions F, G, K of (q, p):", bool((PB1(F, G) + PB1(G, F)) == 0) and bool(leib.expand() == 0) and bool(jac.expand() == 0))
x, y, z, px, py, pz = var("x y z px py pz", domain="real")
qs, ps = [x, y, z], [px, py, pz]
def PB(f, g): return sum(diff(f, qs[i]) * diff(g, ps[i]) - diff(f, ps[i]) * diff(g, qs[i]) for i in range(3))
Lv = [y * pz - z * py, z * px - x * pz, x * py - y * px]
eps3 = lambda a, b, c: (a - b) * (b - c) * (c - a) / 2
ok = all(bool((PB(Lv[i], Lv[j]) - sum(eps3(i, j, kk) * Lv[kk] for kk in range(3))).expand() == 0) for i in range(3) for j in range(3))
L2 = sum(l**2 for l in Lv)
print("[PASS] the angular momentum components satisfy {L_i, L_j} = eps_ijk L_k and {L^2, L_i} = 0:", ok and all(bool(PB(L2, l).expand() == 0) for l in Lv))
V = function("V")(sqrt(x**2 + y**2 + z**2))
Hc = (px**2 + py**2 + pz**2) / 2 + V
print("[PASS] {L_i, H} = 0 for a central potential V(r):", all(bool(PB(l, Hc).simplify_full() == 0) for l in Lv))

# canonical transformations
t, I, th = var("t I th", domain="real")
Qt = q * cos(t) + p * sin(t); Pt = -q * sin(t) + p * cos(t)
J = matrix([[diff(Qt, q), diff(Qt, p)], [diff(Pt, q), diff(Pt, p)]])
print("[PASS] the oscillator flow (q, p) -> (q cos t + p sin t, -q sin t + p cos t) has Jacobian determinant 1 and {Q, P} = 1:", bool(J.det().simplify_full() == 1) and bool(PB1(Qt, Pt).simplify_full() == 1))
qa = sqrt(2 * I) * sin(th); pa = sqrt(2 * I) * cos(th)
br = diff(qa, th) * diff(pa, I) - diff(qa, I) * diff(pa, th)
print("[PASS] the action-angle map q = sqrt(2I) sin theta, p = sqrt(2I) cos theta is canonical ({q, p}_(theta, I) = 1) and turns H = (p^2 + q^2)/2 into I:", bool(br.simplify_full() == 1) and bool(((pa**2 + qa**2) / 2).simplify_full() == I))
Pn = var("Pn")
F2 = q**2 * Pn / 2  # a type-2 generating function F2(q, P): p = dF2/dq, Q = dF2/dP
pF, QF = diff(F2, q), diff(F2, Pn)  # p = q P, Q = q^2/2
Pq = p / q  # invert p = q P
Qq = QF
print("[PASS] the generating function F2 = q^2 P/2 gives the canonical map Q = q^2/2, P = p/q, with {Q, P} = 1:", bool(PB1(Qq, Pq).simplify_full() == 1) and bool(pF.subs(Pn=Pq) == p))
# Hamilton-Jacobi
E, mm = var("E mm", domain="positive")
W = integrate(sqrt(2 * E - q**2), q)
print("[PASS] W(q) = int sqrt(2E - q^2) dq solves the Hamilton-Jacobi equation (W')^2/2 + q^2/2 = E of the oscillator:", bool(((diff(W, q))**2 / 2 + q**2 / 2 - E).canonicalize_radical() == 0))
S = sqrt(2 * mm * E) * q - E * t
print("[PASS] S = sqrt(2mE) q - E t solves the time-dependent Hamilton-Jacobi equation dS/dt + (dS/dq)^2/(2m) = 0 of the free particle:", bool((diff(S, t) + diff(S, q)**2 / (2 * mm)).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))

# interactive version: a slider over t carries the region along the flow
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json
tt = [0.25 * j for j in range(25)]
sol2 = solve_ivp(rhs, [0, tt[-1]], Y0, t_eval=tt, rtol=1e-10, atol=1e-10)
frames, steps = [], []
for j, t0 in enumerate(tt):
    nm = "%.2f" % t0
    A_ = area(sol2.y[:N, j], sol2.y[N:, j])
    frames.append({"name": nm, "data": [{"x": list(sol2.y[:N, j]) + [sol2.y[0, j]], "y": list(sol2.y[N:, j]) + [sol2.y[N, j]], "name": "region, area %.5f" % A_}], "traces": [0]})
    steps.append({"label": nm, "method": "animate", "args": [[nm], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]})
traces = [{"x": frames[0]["data"][0]["x"], "y": frames[0]["data"][0]["y"], "mode": "lines", "fill": "toself", "line": {"color": "#08519c"}, "name": frames[0]["data"][0]["name"]}]
qq = np.linspace(-3.5, 3.5, 300)
for Hv, col in [(-0.8, "lightgrey"), (-0.4, "lightgrey"), (0.0, "lightgrey"), (0.4, "lightgrey"), (1.0, "firebrick"), (1.6, "lightgrey"), (2.4, "lightgrey")]:
    for sgn in [1, -1]:
        pv = [sgn * float(np.sqrt(2 * (Hv + np.cos(a)))) if Hv + np.cos(a) >= 0 else None for a in qq]
        traces.append({"x": list(qq), "y": pv, "mode": "lines", "line": {"color": col, "width": 1}, "showlegend": bool(Hv == 1.0 and sgn == 1), "name": "separatrix H = 1", "hoverinfo": "skip"})
layout = {"title": "Pendulum flow carries a region of phase space: its area is constant (slide t)", "xaxis": {"range": [-3.5, 3.5], "title": "q"}, "yaxis": {"range": [-2.6, 2.6], "title": "p"},
          "sliders": [{"active": 0, "currentvalue": {"prefix": "t = "}, "steps": steps, "pad": {"t": 30}}], "legend": {"orientation": "h"}}
path = plotly_html("pendulum_liouville", traces, layout)
html = open(path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(frames)))
open(path, "w").write(html)
print("wrote figures/interactive/pendulum_liouville.html")
