# Figure: the inner products of the sphere's Killing vectors enter the reduced curvature scalar,
# R_6 = R_4 - 2 + (1/4) (K_a . K_b) F^a F^b (signature (+,-,-,-,-,-): the internal block is h = -gamma with gamma the
# round metric, so R_h = -2, and K_a . K_b = gamma_{mn} K^m_a K^n_b). Pointwise K_a . K_b = delta_ab - n_a n_b varies over the sphere (left:
# K_1 . K_1 = 1 - sin^2 th cos^2 ph as a map), while its average over the unit sphere is (2/3) delta_ab (right: the
# averages, checked by integration). SageManifolds computes R_6 for A^1_x = c t, where R_6 = -2 - (c^2/2) K_1 . K_1
# is not constant on the sphere: the ansatz does not solve the vacuum equations.
# Output: figures/killing_inner_products.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "killing_inner_products.svg")
th, ph = var("th ph")
K = [vector([-sin(ph), -cot(th) * cos(ph)]), vector([cos(ph), -cot(th) * sin(ph)]), vector([0, 1])]
h = diagonal_matrix([1, sin(th)^2])
G = matrix(SR, 3, 3, lambda a, b: (K[a] * h * K[b]).simplify_full())
avg = matrix(QQ, 3, 3, lambda a, b: integrate(integrate(G[a, b] * sin(th), th, 0, pi), ph, 0, 2 * pi) / (4 * pi))
print("average of K_a . K_b over the unit sphere:"); print(avg)
print("[PASS] the average of K_a . K_b over the unit sphere is (2/3) delta_ab:", avg == 2 / 3 * identity_matrix(QQ, 3))
# the curvature scalar for A^1_x = c t
M = Manifold(6, "M", structure="Lorentzian"); X = M.chart("t x y z th6 ph6"); t, x, y, z, th6, ph6 = X[:]
c = var("c")
K1 = K[0].subs(th=th6, ph=ph6); h6 = h.subs(th=th6)
B = [[0, K1[m] * c * t, 0, 0] for m in range(2)]
eta = [1, -1, -1, -1]
g = M.metric("g")                                            # internal block h = -gamma
for mu in range(4):
    for nu in range(mu, 4): g[mu, nu] = (eta[mu] if mu == nu else 0) - sum(h6[m, n] * B[m][mu] * B[n][nu] for m in range(2) for n in range(2))
    for m in range(2): g[mu, 4 + m] = -sum(h6[m, n] * B[n][mu] for n in range(2))
for m in range(2):
    for n in range(m, 2): g[4 + m, 4 + n] = -h6[m, n]
R6 = g.ricci_scalar().expr().simplify_full()
print("R_6 for A^1_x = c t:", R6)
print("[PASS] R_6 = -2 - (c^2/2) K_1 . K_1 = -2 - (c^2/2)(1 - sin^2 th cos^2 ph), varying over the sphere:", bool((R6 - (-2 - c^2 / 2 * G[0, 0].subs(th=th6, ph=ph6))).simplify_full() == 0))
left = contour_plot(G[0, 0], (ph, 0, 2 * pi), (th, 0, pi), fill=True, cmap="viridis", contours=12, colorbar=True)
left.axes_labels([r"$\varphi$", r"$\theta$"])
right = Graphics()
for a in range(3):
    for b in range(3):
        v = float(avg[a, b])
        right += polygon([(b, -a), (b + 1, -a), (b + 1, -a - 1), (b, -a - 1)], color="royalblue" if v > 0 else "white", alpha=0.6 if v > 0 else 0.1, edgecolor="black")
        right += text("%s" % avg[a, b], (b + 0.5, -a - 0.5), fontsize=10)
right += text(r"$\langle K_a\cdot K_b\rangle_{S^2}$", (1.5, 0.4), fontsize=10)
right.axes(False); right.set_aspect_ratio(1)
fig = graphics_array([left, right]).matplotlib(figsize=[9, 3.6]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))
# interactive version: R_6 = -2 - (c^2/2) K_a . K_a on the sphere for a single gauge field A^a_x = c t (c = 1), a = 1, 2, 3
# (a = 1 is the SageManifolds result above; a = 2, 3 follow from it by the rotations that permute the axes)
print("[PASS] K_1.K_1, K_2.K_2, K_3.K_3 are related by the axis permutations (1 - n_a^2):", all(bool((G[a, a] - (1 - [sin(th) * cos(ph), sin(th) * sin(ph), cos(th)][a]^2)).simplify_full() == 0) for a in range(3)))
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
_u = [k * 2 * float(pi) / 60 for k in range(61)]; _v = [k * float(pi) / 30 for k in range(31)]
_traces = []
for _a in range(3):
    _f = fast_callable(-2 - G[_a, _a] / 2, vars=[th, ph], domain=RDF)
    _traces.append({"type": "surface", "x": [[sin(b) * cos(a) for a in _u] for b in _v], "y": [[sin(b) * sin(a) for a in _u] for b in _v],
                    "z": [[cos(b) for a in _u] for b in _v], "surfacecolor": [[_f(max(b, 1e-6), a) for a in _u] for b in _v],
                    "cmin": -2.5, "cmax": -2.0, "colorscale": "Viridis", "colorbar": {"title": "R_6"}, "visible": _a == 0,
                    "name": "A^%d_x = t" % (_a + 1), "hovertemplate": "R_6 = %{surfacecolor:.3f}<extra></extra>"})
_buttons = [{"label": "A^%d_x = t" % (_a + 1), "method": "update", "args": [{"visible": [_b == _a for _b in range(3)]}]} for _a in range(3)]
plotly_html("killing_inner_products", _traces, {"title": "R_6 = -2 - (1/2) K_a.K_a for one gauge field A^a_x = t (choose a)",
            "updatemenus": [{"buttons": _buttons, "direction": "right", "x": 0.0, "y": 1.08, "xanchor": "left", "type": "buttons"}],
            "scene": {"aspectmode": "data"}, "margin": {"l": 10, "r": 10, "t": 60, "b": 10}})
print("wrote figures/interactive/killing_inner_products.html")
