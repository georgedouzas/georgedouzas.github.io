# Figure: the Bloch sphere of a spin-1/2. Left: the state cos(theta/2)|up> + e^(i phi) sin(theta/2)|down> is the
# point n = (sin th cos ph, sin th sin ph, cos th) = <sigma>; under H = (hbar omega/2) sigma_z the arrow precesses
# about the z-axis along the dashed circle (theta = pi/3, phi = pi/4 at t = 0). Right: the three components of
# <sigma>(t). Checks: the Pauli algebra, spin matrices for s = 1/2..2 with hbar symbolic, the 2 pi rotation sign,
# the SU(2) exponential, precession, Clebsch-Gordan coefficients and the two-spin coupling spectrum.
# Output: figures/qm_bloch_sphere.svg (and an interactive threejs sphere).
import os
from sage.functions.wigner import clebsch_gordan
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm_bloch_sphere.svg")
hbar = var("hbar", domain="positive"); t, w, th_ = var("t w th_", domain="real")
s0 = identity_matrix(SR, 2)
sig = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
eps = lambda i, j, k: sign(Permutation([i + 1, j + 1, k + 1])) if len({i, j, k}) == 3 else 0
print("[PASS] the Pauli algebra sigma_i sigma_j = delta_ij 1 + i eps_ijk sigma_k for all i, j:",
      all(sig[i] * sig[j] == (1 if i == j else 0) * s0 + I * sum(eps(i, j, k) * sig[k] for k in range(3)) for i in range(3) for j in range(3)))
def spin(s):
    ms = [s - k for k in range(int(2 * s) + 1)]; d = len(ms)
    Jz = hbar * diagonal_matrix(SR, ms)
    Jp = matrix(SR, d, d, lambda i, j: hbar * sqrt(s * (s + 1) - ms[j] * (ms[j] + 1)) if i == j - 1 else 0)
    Jm = Jp.conjugate_transpose()
    return (Jp + Jm) / 2, (Jp - Jm) / (2 * I), Jz
ok = True
for s in [1 / 2, 1, 3 / 2, 2]:
    Jx, Jy, Jz = spin(s)
    d = Jz.nrows()
    ok = ok and (Jx * Jy - Jy * Jx - I * hbar * Jz).simplify_full().is_zero() and (Jy * Jz - Jz * Jy - I * hbar * Jx).simplify_full().is_zero() \
        and (Jz * Jx - Jx * Jz - I * hbar * Jy).simplify_full().is_zero() and (Jx ** 2 + Jy ** 2 + Jz ** 2 - hbar ** 2 * s * (s + 1) * identity_matrix(d)).simplify_full().is_zero()
print("[PASS] the spin-s matrices (s = 1/2, 1, 3/2, 2) satisfy [J_x, J_y] = i hbar J_z (cyclic) and J^2 = hbar^2 s(s + 1):", ok)
print("[PASS] the rotation by 2 pi, exp(-2 pi i J_z/hbar), is (-1)^(2s) (s = 1/2, 1, 3/2, 2):",
      all((-2 * pi * I * spin(s)[2] / hbar).exp().simplify_full() == (-1) ** int(2 * s) * identity_matrix(int(2 * s) + 1) for s in [1 / 2, 1, 3 / 2, 2]))
U = (-I * th_ * sig[0] / 2).exp()
print("[PASS] exp(-i theta sigma_x/2) = cos(theta/2) 1 - i sin(theta/2) sigma_x:", (U - (cos(th_ / 2) * s0 - I * sin(th_ / 2) * sig[0])).apply_map(lambda e: e.exponentialize().simplify_full()).is_zero())
th, ph = pi / 3, pi / 4
psi0 = vector(SR, [cos(th / 2), exp(I * ph) * sin(th / 2)])
Ut = (-I * w * t * sig[2] / 2).exp()
psit = Ut * psi0
bl = [(psit.conjugate() * S * psit).rectform().trig_reduce().simplify_full() for S in sig]
exp_bl = [sin(th) * cos(ph + w * t), sin(th) * sin(ph + w * t), cos(th)]
print("[PASS] under H = (hbar omega/2) sigma_z the Bloch vector precesses: <sigma>(t) = (sin th cos(ph + omega t), sin th sin(ph + omega t), cos th):",
      all(bool((b - e).trig_expand().simplify_full() == 0) for b, e in zip(bl, exp_bl)))
cg = clebsch_gordan(1, 1 / 2, 3 / 2, 1, -1 / 2, 1 / 2), clebsch_gordan(1, 1 / 2, 3 / 2, 0, 1 / 2, 1 / 2), clebsch_gordan(1, 1 / 2, 1 / 2, 1, -1 / 2, 1 / 2), clebsch_gordan(1, 1 / 2, 1 / 2, 0, 1 / 2, 1 / 2)
print("[PASS] Clebsch-Gordan for 1 x 1/2: |3/2, 1/2> = sqrt(1/3)|1, -1/2> + sqrt(2/3)|0, 1/2>, |1/2, 1/2> = sqrt(2/3)|1, -1/2> - sqrt(1/3)|0, 1/2>:",
      bool(cg[0] == sqrt(1 / 3)) and bool(cg[1] == sqrt(2 / 3)) and bool(cg[2] == sqrt(2 / 3)) and bool(cg[3] == -sqrt(1 / 3)))
def cg_matrix(j1, j2):
    rows = [(m1, m2) for m1 in [j1 - k for k in range(int(2 * j1) + 1)] for m2 in [j2 - k for k in range(int(2 * j2) + 1)]]
    cols = [(j, mm) for j in [j1 + j2 - k for k in range(int(2 * min(j1, j2)) + 1)] for mm in [j - k for k in range(int(2 * j) + 1)]]
    return matrix(SR, len(rows), len(cols), lambda r, c: clebsch_gordan(j1, j2, cols[c][0], rows[r][0], rows[r][1], cols[c][1]))
print("[PASS] the Clebsch-Gordan matrices of 1/2 x 1/2, 1 x 1/2, 1 x 1 and 3/2 x 1 are orthogonal (C^T C = 1 to 1e-14):",
      all(max(abs(z) for z in (cg_matrix(a_, b_).transpose() * cg_matrix(a_, b_) - identity_matrix(cg_matrix(a_, b_).nrows())).n().list()) < 1e-14 for a_, b_ in [(1 / 2, 1 / 2), (1, 1 / 2), (1, 1), (3 / 2, 1)]))
print("[PASS] dimension count: sum over j = |j1 - j2|..j1 + j2 of (2j + 1) = (2 j1 + 1)(2 j2 + 1) for 2 j1, 2 j2 <= 6:",
      all(sum(2 * (abs(a_ - b_) + k) + 1 for k in range(int(a_ + b_ - abs(a_ - b_)) + 1)) == (2 * a_ + 1) * (2 * b_ + 1) for a_ in [k / 2 for k in range(7)] for b_ in [k / 2 for k in range(7)]))
S1 = [hbar / 2 * S.tensor_product(s0) for S in sig]; S2 = [hbar / 2 * s0.tensor_product(S) for S in sig]
SS = sum(a_ * b_ for a_, b_ in zip(S1, S2))
print("[PASS] for two spins 1/2, S1.S2 has eigenvalues hbar^2/4 (three times, the triplet) and -3 hbar^2/4 (once, the singlet):",
      sorted(SS.charpoly().roots(multiplicities=True), key=lambda r: r[1]) == [(-3 * hbar ** 2 / 4, 1), (hbar ** 2 / 4, 3)])
# the figure: Bloch sphere in orthographic projection, and the components of <sigma>(t)
az, el = -0.6, 0.35
def P(X, Y, Z):
    return (-X * sin(az) + Y * cos(az), -(X * cos(az) + Y * sin(az)) * sin(el) + Z * cos(el))
u = var("u")
g1 = circle((0, 0), 1, color="black", thickness=1.2)
g1 += parametric_plot(P(cos(u), sin(u), 0), (u, 0, 2 * pi), color="gray", thickness=0.8)
g1 += parametric_plot(P(sin(u), 0, cos(u)), (u, 0, 2 * pi), color="lightgray", thickness=0.8)
for vec, lab in [((1.25, 0, 0), "x"), ((0, 1.25, 0), "y"), ((0, 0, 1.25), "z")]:
    g1 += arrow((0, 0), P(*vec), color="black", width=0.8, arrowsize=2) + text("$%s$" % lab, P(*[1.12 * c for c in vec]), fontsize=12, color="black")
g1 += parametric_plot(P(sin(th) * cos(u), sin(th) * sin(u), cos(th)), (u, 0, 2 * pi), color="firebrick", linestyle="--", thickness=1)
nb = (sin(th) * cos(ph), sin(th) * sin(ph), cos(th))
g1 += arrow((0, 0), P(*nb), color="firebrick", width=1.8, arrowsize=3) + text(r"$\langle\vec\sigma\rangle$", P(*[1.18 * c for c in nb]), fontsize=12, color="firebrick")
g1 += point(P(0, 0, 1), size=25, color="royalblue") + text(r"$|{\uparrow}\rangle$", (P(0, 0, 1)[0] + 0.22, P(0, 0, 1)[1] + 0.02), fontsize=11, color="royalblue")
g1 += point(P(0, 0, -1), size=25, color="royalblue") + text(r"$|{\downarrow}\rangle$", (P(0, 0, -1)[0] + 0.22, P(0, 0, -1)[1] - 0.05), fontsize=11, color="royalblue")
g1.axes(False); g1.set_aspect_ratio(1)
ws = var("ws")
g2 = plot(sin(th) * cos(ph + ws), (ws, 0, 4 * pi), color="firebrick", thickness=1.6, legend_label=r"$\langle\sigma_x\rangle$")
g2 += plot(sin(th) * sin(ph + ws), (ws, 0, 4 * pi), color="darkgreen", thickness=1.6, legend_label=r"$\langle\sigma_y\rangle$")
g2 += plot(cos(th) + 0 * ws, (ws, 0, 4 * pi), color="royalblue", thickness=1.6, legend_label=r"$\langle\sigma_z\rangle$")
g2.axes_labels([r"$\omega t$", ""]); g2.set_legend_options(font_size=9, loc="lower center", ncol=3); g2.set_axes_range(ymin=-1.45, ymax=1.05)
G = graphics_array([g1, g2])
G.matplotlib(figsize=[10, 4.2]).savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import threejs_html
a_, b_ = var("a_ b_")
sc = parametric_plot3d((sin(a_) * cos(b_), sin(a_) * sin(b_), cos(a_)), (a_, 0, pi), (b_, 0, 2 * pi), color="lightblue", opacity=0.25, mesh=False)
sc += parametric_plot3d((cos(u), sin(u), 0), (u, 0, 2 * pi), color="gray", thickness=1)
for vec, c in [((1.3, 0, 0), "black"), ((0, 1.3, 0), "black"), ((0, 0, 1.3), "black")]:
    sc += arrow3d((0, 0, 0), vec, color=c, width=1)
sc += text3d("x", (1.45, 0, 0)) + text3d("y", (0, 1.45, 0)) + text3d("z  |up>", (0, 0, 1.45)) + text3d("|down>", (0, 0, -1.2))
sc += parametric_plot3d((sin(th) * cos(u), sin(th) * sin(u), cos(th)), (u, 0, 2 * pi), color="firebrick", thickness=2)
for k in range(6):
    phk = ph + k * pi / 3
    sc += arrow3d((0, 0, 0), (sin(th) * cos(phk), sin(th) * sin(phk), cos(th)), color="firebrick" if k == 0 else "salmon", width=2 if k == 0 else 1)
for (tq, pq, lab) in [(pi / 2, 0, "|+x>"), (pi / 2, pi / 2, "|+y>")]:
    sc += point3d((sin(tq) * cos(pq), sin(tq) * sin(pq), cos(tq)), size=12, color="royalblue") + text3d(lab, (1.15 * sin(tq) * cos(pq), 1.15 * sin(tq) * sin(pq), 0.12))
threejs_html("qm_bloch_sphere", sc)
print("wrote figures/interactive/qm_bloch_sphere.html")
