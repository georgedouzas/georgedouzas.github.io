# Figure: the number of gauge bosons of a Kaluza-Klein reduction is the dimension of the isometry group of the internal
# space, not its dimension. Bars: dim M_k (gray) and dim Isom(M_k) (blue) for a list of internal spaces, with the
# maximal value k(k+1)/2 of a maximally symmetric space (black ticks). Checked here: the rotation fields of S^n are
# n(n+1)/2 independent tangent fields (n = 1..6); the three Killing vectors of S^2; on the round S^3 = SU(2) the three
# left- and three right-invariant fields are six independent commuting families of Killing fields; on the squashed
# (Berger) S^3 the right-invariant fields and e_3 stay Killing while e_1, e_2 do not. The remaining entries of the
# chart (CP^2, S^7, squashed S^7, SU(3), Witten's product space) are quoted.
# Output: figures/nak_isometry_count.svg (+ interactive version).
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nak_isometry_count.svg")
# (1) rotation fields of S^n
d = lambda a, b: 1 if a == b else 0
ok = True
for n in range(1, 7):
    x = vector(var(" ".join("x%d" % i for i in range(n + 1))))
    fields = [vector([x[i] * d(j, k) - x[j] * d(i, k) for k in range(n + 1)]) for i in range(n + 1) for j in range(i + 1, n + 1)]
    coeffs = matrix(QQ, [[f[k].coefficient(x[l]) for k in range(n + 1) for l in range(n + 1)] for f in fields])
    ok &= coeffs.rank() == n * (n + 1) / 2 and all(bool((x * f).expand() == 0) for f in fields)
print("[PASS] the rotation fields x_i d_j - x_j d_i of S^n are n(n+1)/2 independent tangent fields, n = 1, ..., 6:", ok)
# (2) the sphere S^2
th, ph = var("th ph")
K = [vector([-sin(ph), -cot(th) * cos(ph)]), vector([cos(ph), -cot(th) * sin(ph)]), vector([0, 1])]
gam = diagonal_matrix(SR, [1, sin(th)^2])
def lie2(Xv, m):
    Y = [th, ph]
    return matrix(SR, 2, 2, lambda i, j: sum(Xv[k] * diff(m[i, j], Y[k]) + diff(Xv[k], Y[i]) * m[k, j] + diff(Xv[k], Y[j]) * m[i, k] for k in range(2)))
print("[PASS] S^2: the three fields K_a are Killing (isometry algebra so(3), dimension 3):", all(lie2(Kv, gam).apply_map(lambda e: e.simplify_full()).is_zero() for Kv in K))
# (3) S^3 = SU(2) in Euler angles, metric a^2 (s1^2 + s2^2) + b^2 s3^2
a, b = var("a b"); assume(a > 0, b > 0)
M = Manifold(3, "S3", structure="Riemannian")
X.<t3, p3, s3> = M.chart(r"t3:(0,pi):\theta p3:(0,2*pi):\varphi s3:(0,4*pi):\psi")
sL = [vector([cos(s3), sin(t3) * sin(s3), 0]), vector([-sin(s3), sin(t3) * cos(s3), 0]), vector([0, cos(t3), 1])]
sR = [vector([cos(p3), 0, sin(t3) * sin(p3)]), vector([-sin(p3), 0, sin(t3) * cos(p3)]), vector([0, 1, cos(t3)])]
E = matrix(SR, [v.list() for v in sL]).inverse().apply_map(lambda e: e.simplify_full())
XI = matrix(SR, [v.list() for v in sR]).inverse().apply_map(lambda e: e.simplify_full())
e = [M.vector_field(*E.column(k).list()) for k in range(3)]; xi = [M.vector_field(*XI.column(k).list()) for k in range(3)]
def metric(aa, bb, name):
    G = sum(r^2 * v.column() * v.row() for r, v in zip([aa, aa, bb], sL)); g = M.metric(name)
    for i in range(3):
        for j in range(i, 3): g[i, j] = G[i, j].simplify_full()
    return g
iszero = lambda T: all(bool(T[i, j].expr().simplify_full() == 0) for i in range(3) for j in range(3))
round_ = metric(1 / 2, 1 / 2, "gr")
br_zero = all(all(bool(c.expr().simplify_full() == 0) for c in (e[i].bracket(xi[j])).comp()[:]) for i in range(3) for j in range(3))
six = matrix(SR, [E.column(k).list() for k in range(3)] + [XI.column(k).list() for k in range(3)])
pts = [(pi / 2, 0, 0), (pi / 2, pi / 2, 0), (pi / 2, 0, pi / 2), (pi / 2, pi / 2, pi / 2)]       # trigonometric values in {0, 1, -1}
stack = matrix(QQ, [[QQ(c.subs(t3=v1, p3=v2, s3=v3)) for c in r] for (v1, v2, v3) in pts for r in six.columns()])
indep = stack.rank() == 6                          # as functions the six fields are independent over R
print("[PASS] round S^3: the left-invariant e_a and right-invariant xi_a are six Killing fields, [e_a, xi_b] = 0, independent over R:",
      all(iszero(round_.lie_derivative(v)) for v in e + xi) and br_zero and indep)
berger = metric(a, b, "gb")
print("[PASS] Berger S^3 (a != b): xi_1, xi_2, xi_3 and e_3 are Killing, e_1 and e_2 are not (isometry algebra su(2) + u(1), dimension 4):",
      all(iszero(berger.lie_derivative(v)) for v in xi + [e[2]]) and not iszero(berger.lie_derivative(e[0])) and not iszero(berger.lie_derivative(e[1])))
# the chart
spaces = [(r"$S^1$", 1, 1), (r"$T^2$", 2, 2), (r"$S^2$", 2, 3), (r"$S^3$", 3, 6), (r"squashed $S^3$", 3, 4), (r"$\mathbb{CP}^2$", 4, 8),
          (r"$S^4$", 4, 10), (r"$\mathbb{CP}^2\times S^2\times S^1$", 7, 12), (r"squashed $S^7$", 7, 13), (r"$S^7$", 7, 28), (r"$SU(3)$", 8, 16)]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 3.8))
xs = list(range(len(spaces)))
ax.bar([i - 0.2 for i in xs], [s[1] for s in spaces], width=0.4, color="0.7", label=r"$\dim M_k$")
ax.bar([i + 0.2 for i in xs], [s[2] for s in spaces], width=0.4, color="royalblue", label=r"$\dim\,\mathrm{Isom}(M_k)$ = number of gauge bosons")
for i, s in enumerate(spaces):
    kk = s[1] * (s[1] + 1) / 2
    ax.plot([i - 0.4, i + 0.4], [kk, kk], color="black", lw=1.2)
    ax.text(i + 0.2, s[2] + 0.4, str(s[2]), ha="center", fontsize=8)
ax.plot([], [], color="black", lw=1.2, label=r"maximum $k(k+1)/2$")
ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in spaces], rotation=25, ha="right", fontsize=9)
ax.set_ylim(0, 38); ax.legend(fontsize=8, loc="upper left"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
_names = ["S^1", "T^2", "S^2", "S^3 (round)", "S^3 (squashed)", "CP^2", "S^4", "CP^2 x S^2 x S^1", "S^7 (squashed)", "S^7 (round)", "SU(3)"]
_groups = ["U(1)", "U(1)^2", "SO(3)", "SO(4)", "SU(2) x U(1)", "SU(3)", "SO(5)", "SU(3) x SU(2) x U(1)", "Sp(2) x Sp(1)", "SO(8)", "SU(3) x SU(3)"]
_kk = [k * 0.25 for k in range(0, 37)]
_traces = [{"x": _kk, "y": [k * (k + 1) / 2 for k in _kk], "mode": "lines", "name": "maximum k(k+1)/2", "line": {"color": "black", "dash": "dot"}},
           {"x": [s[1] for s in spaces], "y": [s[2] for s in spaces], "mode": "markers+text", "name": "internal spaces",
            "text": _names, "textposition": "middle right", "marker": {"size": 10, "color": "royalblue"},
            "customdata": _groups, "hovertemplate": "%{text}: k = %{x}, isometry group %{customdata}, %{y} gauge bosons<extra></extra>"}]
plotly_html("nak_isometry_count", _traces, {"title": "Gauge bosons = dim of the isometry group (hover for the group)",
            "xaxis": {"title": "dimension k of the internal space", "range": [0, 9.5]}, "yaxis": {"title": "number of gauge bosons", "range": [0, 38]}})
print("wrote figures/interactive/nak_isometry_count.html")
