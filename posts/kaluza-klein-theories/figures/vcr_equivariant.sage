# Figure: two symmetric-field reductions over the sphere. Left: the Higgs potential of SU(2)-equivariant reduction of
# Dolan-Szabo with m = 1 (one bifundamental Higgs field phi between blocks of rank k0 >= k1), as a function of the
# radial value r = |phi| along the vacuum direction, in units g = R = 1: V = g^2 (r^2 - c)^2 + (k0 - k1) g^2 c^2 / 2 with
# c = 1/(4 g^2 R^2); the minimum sits at r = 1/(2 g R) and its height is (k0 - k1)/(32 g^2 R^4). Right: the fuzzy
# sphere: for the N-dimensional irreducible representation of su(2) the eigenvalues of X_3 = J_3 / sqrt(j(j+1))
# are N latitudes on the unit sphere, which fill it as N grows. Output: figures/vcr_equivariant.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vcr_equivariant.svg")
r = var("r")
c0 = 1 / 4
left = Graphics()
for (k0, k1), col in [((1, 1), "royalblue"), ((2, 1), "firebrick"), ((3, 1), "darkgreen")]:
    left += plot((r ** 2 - c0) ** 2 + (k0 - k1) * c0 ** 2 / 2, (r, 0, 0.85), color=col, thickness=2, legend_label="$(k_0, k_1) = (%d, %d)$" % (k0, k1))
left += line([(0.5, 0), (0.5, 0.12)], color="gray", linestyle="--")
left.axes_labels([r"$|\phi|$ ($g = R = 1$)", r"$V$"]); left.set_legend_options(loc="upper center", font_size=8); left.ymax(0.14)
right = Graphics()
for k, N in enumerate([2, 3, 5, 8]):
    j = QQ(N - 1) / 2
    xo = 2.4 * k
    right += circle((xo, 0), 1, color="lightgray")
    for mval in [j - i for i in range(N)]:
        zz = float(mval / sqrt(j * (j + 1)))
        hw = sqrt(max(0, 1 - zz ** 2))
        right += line([(xo - hw, zz), (xo + hw, zz)], color="royalblue", thickness=2)
    right += text("$N=%d$" % N, (xo, -1.25), fontsize=9)
right.axes(False); right.set_aspect_ratio(1)
left._extra_kwds.update(frame=True, axes=False)
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[11, 3.8]); fig.savefig(OUT, bbox_inches="tight")

gg, R = var("g R", domain="positive")
c = 1 / (4 * gg ** 2 * R ** 2)
def V_m1(phi, k0, k1):
    P = phi * phi.conjugate_transpose(); Q = phi.conjugate_transpose() * phi
    return (gg ** 2 / 2 * ((c * identity_matrix(k0) - P) ** 2).trace() + gg ** 2 / 2 * ((Q - c * identity_matrix(k1)) ** 2).trace())
u = var(" ".join("u%d" % i for i in range(24)), domain="real")
def phimat(k0, k1):
    return matrix(SR, k0, k1, lambda i, j: u[2 * (i * k1 + j)] + I * u[2 * (i * k1 + j) + 1])
ok = True
for k0, k1 in [(1, 1), (2, 1), (3, 1), (2, 2), (3, 2)]:
    ph = phimat(k0, k1)
    Q = ph.conjugate_transpose() * ph
    lhs = V_m1(ph, k0, k1).expand()
    rhs = ((k0 - k1) * gg ** 2 * c ** 2 / 2 + gg ** 2 * ((c * identity_matrix(k1) - Q) ** 2).trace()).expand()
    ok = ok and bool((lhs - rhs).simplify_full() == 0)
    vac = matrix(SR, k0, k1, lambda i, j: (1 / (2 * gg * R)) if i - (k0 - k1) == j else 0)
    ok = ok and bool((V_m1(vac, k0, k1) - (k0 - k1) / (32 * gg ** 2 * R ** 4)).simplify_full() == 0)
print("[PASS] equivariant m = 1 potential: V = (g^2/2) tr(c - phi phi^+)^2 + (g^2/2) tr(phi^+ phi - c)^2 = (k0-k1) g^2 c^2/2 + g^2 tr(c - phi^+ phi)^2, c = 1/(4 g^2 R^2), with V = (k0 - k1)/(32 g^2 R^4) at phi^0 = (0; 1/(2gR)), for (k0,k1) = (1,1), (2,1), (3,1), (2,2), (3,2):", ok)
v, h, chi = var("v h chi", domain="real")
V11 = gg ** 2 * ((v + h / sqrt(2)) ** 2 + chi ** 2 / 2 - c) ** 2
quad = V11.subs(v=sqrt(c)).taylor((h, 0), (chi, 0), 2)
print("[PASS] (k0, k1) = (1, 1): V = g^2(|phi|^2 - c)^2 is minimal at |phi| = 1/(2gR) with V = 0; expanding phi = v + (h + i chi)/sqrt(2) gives the Higgs mass mu_h = 1/R and a massless Goldstone chi:",
      bool(diff(gg ** 2 * (r ** 2 - c) ** 2, r).subs(r=1 / (2 * gg * R)) == 0) and bool((quad - h ** 2 / (2 * R ** 2)).simplify_full() == 0))
M2 = matrix(SR, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -sqrt(3)], [0, 0, -sqrt(3), 3]]) / (8 * R ** 2)
ev = sorted(M2.eigenvalues(), key=lambda e: e.subs(R=1))
null = M2.right_kernel().basis()[0]
print("[PASS] (k0, k1) = (2, 1), SU(3) -> SU(2) x U(1) -> U(1): the vector mass matrix (Dolan-Szabo eq. 4.17) has eigenvalues 0, 1/(8R^2) twice, 1/(2R^2), so mu_W = 1/(2 sqrt 2 R), mu_Z = 2 mu_W, photon (sqrt 3 W^3 + B)/2 and sin^2 theta_W = 3/4:",
      bool(ev[0] == 0) and bool(ev[1] == 1 / (8 * R ** 2)) and bool(ev[2] == 1 / (8 * R ** 2)) and bool(ev[3] == 1 / (2 * R ** 2))
      and bool(null[3] / null[2] == 1 / sqrt(3)) and bool(1 - (sqrt(ev[1]) / sqrt(ev[3])) ** 2 == QQ(3) / 4))
okm = True
for mm_ in range(1, 6):
    Sigma = diagonal_matrix(SR, [mm_ - 2 * i for i in range(mm_ + 1)])
    phi0 = matrix(SR, mm_ + 1, mm_ + 1, lambda i, j: sqrt((i + 1) * (mm_ - i)) / (2 * gg * R) if j == i + 1 else 0)
    okm = okm and bool(((phi0 * phi0.conjugate_transpose() - phi0.conjugate_transpose() * phi0) - Sigma / (4 * gg ** 2 * R ** 2)).simplify_full() == 0)
print("[PASS] equal-rank vacua (Dolan-Szabo eq. 3.11, n = 1): phi_i^0 = sqrt(i(m - i + 1))/(2gR) solves [phi, phi^+] = Sigma/(4 g^2 R^2), Sigma = diag(m, m-2, ..., -m), for m = 1, ..., 5:", okm)
def spin(N):
    j = QQ(N - 1) / 2; ms = [j - k for k in range(N)]
    Jp = matrix(QQbar, N, N, lambda a, b: sqrt(j * (j + 1) - ms[b] * (ms[b] + 1)) if a == b - 1 else 0)
    Jm = Jp.transpose(); J3 = diagonal_matrix(QQbar, ms)
    return [(Jp + Jm) / 2, (Jp - Jm) / (2 * QQbar(I)), J3], j
okf = True
for N in range(2, 7):
    J, j = spin(N)
    eps = lambda a, b, cc_: sign(Permutation([a + 1, b + 1, cc_ + 1])) if len({a, b, cc_}) == 3 else 0
    okf = okf and all(J[a] * J[b] - J[b] * J[a] == QQbar(I) * sum(eps(a, b, k) * J[k] for k in range(3)) for a in range(3) for b in range(3))
    okf = okf and sum(Ji * Ji for Ji in J) == (QQ(N ** 2 - 1) / 4) * identity_matrix(QQbar, N)
    F = [[J[a] * J[b] - J[b] * J[a] - QQbar(I) * sum(eps(a, b, k) * J[k] for k in range(3)) for b in range(3)] for a in range(3)]
    okf = okf and sum((F[a][b].conjugate_transpose() * F[a][b]).trace() for a in range(3) for b in range(3)) == 0
print("[PASS] fuzzy sphere: the spin-j matrices of dimension N = 2j + 1 satisfy [J_a, J_b] = i eps_abc J_c and J_a J_a = (N^2 - 1)/4, so F_ab = [phi_a, phi_b] - i eps_abc phi_c and V = (1/4) tr F^+ F vanish at phi_a = J_a, for N = 2, ..., 6:", okf)
print("wrote", os.path.relpath(OUT))

# Interactive version: V(|phi|) for (k0, k1) = (1, 1), (2, 1), (3, 1) with a slider for gR (g = 1): the minimum moves
# to |phi| = 1/(2gR) and the vacuum energy (k0 - k1)/(32 g^2 R^4) falls as R grows.
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json
xs = [1.2 * k / 200 for k in range(201)]
pairs = [(1, 1), (2, 1), (3, 1)]
def _data(Rv):
    cv = 1 / (4 * Rv ** 2)
    return [{"x": xs, "y": [(x0 ** 2 - cv) ** 2 + (k0 - k1) * cv ** 2 / 2 for x0 in xs]} for k0, k1 in pairs]
_vals = [round(0.7 + 0.1 * j, 1) for j in range(0, 14)]
_frames, _steps = [], []
for Rv in _vals:
    _frames.append({"name": "%.1f" % Rv, "data": _data(Rv), "traces": [0, 1, 2]})
    _steps.append({"label": "%.1f" % Rv, "method": "animate", "args": [["%.1f" % Rv], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]})
_d0 = _data(1.0)
_traces = [dict(_d0[i], mode="lines", name="(k0, k1) = (%d, %d)" % pairs[i], line={"color": col, "width": 3}) for i, col in enumerate(["royalblue", "firebrick", "darkgreen"])]
_layout = {"title": "Equivariant reduction, m = 1: V(|phi|) with g = 1; slide the sphere radius R",
           "xaxis": {"title": "|phi|"}, "yaxis": {"title": "V", "range": [0, 0.3]},
           "sliders": [{"active": 3, "currentvalue": {"prefix": "R = "}, "steps": _steps, "pad": {"t": 40}}], "legend": {"orientation": "h", "y": -0.3}}
_path = plotly_html("vcr_equivariant", _traces, _layout)
open(_path, "w").write(open(_path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(_frames))))
print("wrote figures/interactive/vcr_equivariant.html")
