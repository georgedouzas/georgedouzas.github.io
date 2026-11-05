# Figure: the CSDR potential of SU(3) over S^2. From F_12 = [phi_1, phi_2] - tau on the doublet solution,
# V = -Tr(F_12^2) = r^2/2 - r + 2/3 with r = |Phi|^2 = |p|^2 + |q|^2, minimised at r = 1 with V_min = 1/6 (left,
# along a real section; right, over the (Re p, Re q) plane). The vacuum |Phi| = 1 breaks SU(2) x U(1) to U(1)_em.
# Output: figures/csdr_potential.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csdr_potential.svg")
p, q, pb, qb = var("p q pb qb")
E = lambda i, j: matrix(SR, 3, 3, lambda a, b: 1 if (a, b) == (i, j) else 0)
tau = I / 3 * diagonal_matrix(SR, [1, 1, -2])
phip = p * E(0, 2) + q * E(1, 2); phim = -(pb * E(2, 0) + qb * E(2, 1))          # phi_- = -phi_+^dagger with pb = conj p
phi1 = (phip + phim) / 2; phi2 = I * (phip - phim) / 2
F12 = phi1 * phi2 - phi2 * phi1 - tau
V = -(F12 * F12).trace().expand()
r = var("r")
print("V =", V)
print("[PASS] V = r^2/2 - r + 2/3 with r = |p|^2 + |q|^2:", bool((V - ((p * pb + q * qb)^2 / 2 - (p * pb + q * qb) + 2/3)).expand() == 0))
Vr = r^2 / 2 - r + 2/3
print("[PASS] the minimum of V is 1/6 at r = 1 (V = (r-1)^2/2 + 1/6):", bool((Vr - ((r - 1)^2 / 2 + 1/6)).expand() == 0) and solve(diff(Vr, r) == 0, r) == [r == 1])
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
rs = np.linspace(0, 2.4, 300)
ax1.plot(rs, rs**2 / 2 - rs + 2/3, color="firebrick", lw=2, label=r"$V(r) = r^2/2 - r + 2/3$")
ax1.axhline(1/6, color="gray", ls=":", lw=1); ax1.plot([1], [1/6], "o", color="royalblue", ms=8, label=r"minimum $V = 1/6$ at $r = 1$")
ax1.plot([0], [2/3], "s", color="black", ms=6, label=r"$V(0) = 2/3$: the symmetric point")
ax1.set_xlabel(r"$r = |p|^2 + |q|^2$"); ax1.set_ylabel(r"$V$"); ax1.set_ylim(0, 1.3); ax1.set_xlim(0, 2.4); ax1.legend(loc="upper center", fontsize=8)
g = np.linspace(-1.6, 1.6, 241); X, Y = np.meshgrid(g, g); R2 = X**2 + Y**2
Z = np.minimum(R2**2 / 2 - R2 + 2/3, 1.3)
cs = ax2.contourf(X, Y, Z, levels=np.linspace(1/6, 1.3, 15), cmap="viridis")
ax2.add_patch(plt.Circle((0, 0), 1, fill=False, color="firebrick", lw=2)); ax2.set_aspect("equal")
ax2.text(0, 1.08, r"vacua $|\Phi| = 1$", color="white", ha="center", fontsize=9)
ax2.set_xlabel(r"$\mathrm{Re}\,p$"); ax2.set_ylabel(r"$\mathrm{Re}\,q$"); fig.colorbar(cs, ax=ax2, label=r"$V$ (clipped at $1.3$)")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

# the vacuum p = 0, q = 1: residual field strength, unbroken generator, hypercharge, Weinberg angle, gauge boson masses
ad = lambda A, B: A * B - B * A
v1 = phi1.subs({p: 0, q: 1, pb: 0, qb: 1}); v2 = phi2.subs({p: 0, q: 1, pb: 0, qb: 1})
Qem = I * diagonal_matrix(SR, [2, -1, -1]) / 3; T3 = I * diagonal_matrix(SR, [1, -1, 0]) / 2
Fvac = ad(v1, v2) - tau
print("[PASS] at the vacuum F_12 = -Q_em/2 with Q_em = (i/3) diag(2, -1, -1), so V_min = -Tr(Q_em^2)/4 = 1/6:", Fvac == -Qem / 2 and bool(-(Qem * Qem).trace() / 4 == 1/6))
print("[PASS] Q_em = T_3 + tau/2:", Qem == T3 + tau / 2)
# real anti-Hermitian elements of h = s(u(2) + u(1)) annihilating both phi_+ = E_23 and phi_- = -E_32
c1, c2, c3, c4 = var("c1 c2 c3 c4", domain="real")
s1 = matrix(SR, [[0, 1, 0], [1, 0, 0], [0, 0, 0]]); s2 = matrix(SR, [[0, -I, 0], [I, 0, 0], [0, 0, 0]]); s3 = matrix(SR, [[1, 0, 0], [0, -1, 0], [0, 0, 0]])
T = [I / 2 * s1, I / 2 * s2, I / 2 * s3, I / (2 * sqrt(3)) * diagonal_matrix(SR, [1, 1, -2])]
X = c1 * T[0] + c2 * T[1] + c3 * T[2] + c4 * T[3]
eqs = []
for M in (ad(X, E(1, 2)), ad(X, E(2, 1))):
    for e in M.list():
        eqs += [e.real_part(), e.imag_part()]
eqs = [e for e in eqs if bool(e != 0)]
Jm = matrix(SR, [[diff(e, c) for c in (c1, c2, c3, c4)] for e in eqs])
ker = Jm.right_kernel().basis()
print("[PASS] the unbroken subalgebra of h at the vacuum is one-dimensional and spanned by Q_em:", len(ker) == 1 and (sum(ker[0][k] * T[k] for k in range(4)) * (1 / ker[0][2]) - Qem).simplify_full().is_zero())
Yn = diagonal_matrix(SR, [1, 1, -2]) / (2 * sqrt(3)); T3h = diagonal_matrix(SR, [1, -1, 0]) / 2
yd = ad(Yn, E(0, 2))[0, 2]
print("[PASS] Y_n = lambda_8/2 has Tr Y_n^2 = Tr T_3^2 = 1/2 and eigenvalue sqrt(3)/2 on the doublet:", bool((Yn * Yn).trace() == 1/2) and bool((T3h * T3h).trace() == 1/2) and bool(yd == sqrt(3) / 2))
g = var("g"); gp = g * yd / (1/2)
print("[PASS] matching g' Y = g sqrt(3)/2 with Y = 1/2: g' = sqrt(3) g and sin^2 theta_W = g'^2/(g^2 + g'^2) = 3/4:", bool(gp == sqrt(3) * g) and bool((gp^2 / (g^2 + gp^2)).simplify_full() == 3/4))
M2 = matrix(SR, 4, 4, lambda i, j: -2 * sum((ad(T[i], f) * ad(T[j], f)).trace() for f in (v1, v2)))
ev = sorted(M2.eigenvalues())
photon = M2.right_kernel().basis()[0]
print("gauge boson mass matrix (units 1/R^2):", M2.list(), "eigenvalues", ev)
print("[PASS] gauge boson masses^2 (units 1/R^2): photon 0, W 1/2, 1/2, Z 2; the massless direction is Q_em; M_W^2/M_Z^2 = 1/4 = cos^2 theta_W:",
      ev == [0, 1/2, 1/2, 2] and (sum(photon[k] * T[k] for k in range(4)) / photon[2] - Qem).simplify_full().is_zero() and bool(ev[1] / ev[3] == 1 - 3/4))
rho = var("rho")
print("[PASS] along the radial direction q = 1 + rho: V = 1/6 + 2 rho^2 + O(rho^3), so with kinetic term (d rho)^2 the Higgs mass^2 is 2/R^2 = 4 M_W^2:",
      bool(taylor(Vr.subs(r=(1 + rho)^2), rho, 0, 2) == 1/6 + 2 * rho^2) and bool(2 / ev[1] == 4))

# interactive version: the Mexican hat V over the (Re q, Im q) plane (p = 0) and the radial profile V(r)
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
N = 61; grid = [-1.6 + 3.2 * k / (N - 1) for k in range(N)]
Z = [[min(1.2, float((a^2 + b^2)^2 / 2 - (a^2 + b^2) + 2/3)) for a in grid] for b in grid]
th = [2 * float(pi) * k / 120 for k in range(121)]
_traces = [
    {"type": "surface", "x": grid, "y": grid, "z": Z, "colorscale": "Viridis", "showscale": False, "name": "V", "opacity": 0.95,
     "hovertemplate": "Re q = %{x:.2f}<br>Im q = %{y:.2f}<br>V = %{z:.3f}<extra></extra>", "scene": "scene"},
    {"type": "scatter3d", "x": [float(cos(t)) for t in th], "y": [float(sin(t)) for t in th], "z": [1/6] * 121, "mode": "lines",
     "line": {"color": "firebrick", "width": 6}, "name": "vacua |Phi| = 1, V = 1/6", "scene": "scene"},
    {"type": "scatter", "x": [2.6 * k / 200 for k in range(201)], "y": [float((2.6 * k / 200)^2 / 2 - 2.6 * k / 200 + 2/3) for k in range(201)],
     "mode": "lines", "line": {"color": "firebrick", "width": 3}, "name": "V(r) = r^2/2 - r + 2/3", "xaxis": "x", "yaxis": "y"},
    {"type": "scatter", "x": [1], "y": [1/6], "mode": "markers", "marker": {"size": 10, "color": "royalblue"}, "name": "minimum (1, 1/6)", "xaxis": "x", "yaxis": "y"}]
plotly_html("csdr_potential", _traces, {"title": "The CSDR Higgs potential: Mexican hat over (Re q, Im q) at p = 0, and V(r)",
            "scene": {"domain": {"x": [0, 0.58], "y": [0, 1]}, "xaxis": {"title": "Re q"}, "yaxis": {"title": "Im q"}, "zaxis": {"title": "V", "range": [0, 1.2]}},
            "xaxis": {"domain": [0.66, 1], "title": "r = |p|^2 + |q|^2"}, "yaxis": {"title": "V", "range": [0, 1.2]},
            "legend": {"orientation": "h", "y": -0.1}})
print("wrote figures/interactive/csdr_potential.html")
