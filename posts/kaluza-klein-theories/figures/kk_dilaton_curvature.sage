# Figure: the two new terms of the reduced curvature scalar R5 = R4 + (phi^2/4) F^2 - (2/phi) Box phi (signature
# (+,-,-,-,-), Box = d_t^2 - nabla^2), checked with SageManifolds on two five-dimensional metrics with flat
# four-dimensional part. Left: A = 0 and phi = phi(x): Box phi = -phi'' and R5 = 2 phi''/phi. Right: phi = 1 and
# A_x = B t (constant electric field F_{tx} = B, F^2 = -2 B^2): R5 = (1/4) F^2 = -B^2/2.
# Output: figures/kk_dilaton_curvature.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kk_dilaton_curvature.svg")
M = Manifold(5, "M", structure="Lorentzian"); X = M.chart("t x y z w"); t, x, y, z, w = X[:]
eta = [1, -1, -1, -1]
# (a) dilaton only
phi = function("phi")(x)
g = M.metric("g")
for i in range(4): g[i, i] = eta[i]
g[4, 4] = -phi^2
Ra = g.ricci_scalar().expr().simplify_full()
print("R5 for eta - phi(x)^2 dw^2:", Ra)
print("[PASS] with A = 0 and phi = phi(x): R5 = 2 phi''/phi = -(2/phi) Box phi:", bool((Ra - 2 * diff(phi, x, 2) / phi).simplify_full() == 0))
# (b) constant electric field, phi = 1: A_x = B t, F_{tx} = d_t A_x = B
B = var("B")
h = M.metric("h"); Avec = [0, B * t, 0, 0]
for i in range(4):
    for j in range(i, 4): h[i, j] = eta[i] * (1 if i == j else 0) - Avec[i] * Avec[j]
    h[i, 4] = -Avec[i]
h[4, 4] = -1
Rb = h.ricci_scalar().expr().simplify_full()
F2 = 2 * eta[0] * eta[1] * B^2                      # F_{tx} F^{tx} + F_{xt} F^{xt} = 2 eta^tt eta^xx B^2 = -2 B^2
print("R5 for the constant electric field:", Rb, "; (phi^2/4) F^2 =", F2 / 4)
print("[PASS] with phi = 1 and F_{tx} = B: R5 = (1/4) F^2 = -B^2/2:", bool((Rb - F2 / 4).simplify_full() == 0))
xs = var("xs")
phi_ex = 1 + 0.3 * sin(xs)
left = plot(phi_ex, (xs, -6, 6), color="royalblue", thickness=2, legend_label=r"$\phi(x) = 1 + 0.3\sin x$")
left += plot(2 * diff(phi_ex, xs, 2) / phi_ex, (xs, -6, 6), color="firebrick", thickness=2, legend_label=r"$R_5 = 2\phi''/\phi$")
left.axes_labels(["$x$", ""]); left.set_legend_options(loc="lower right", font_size=8)
Bs = var("Bs")
right = plot(-Bs^2 / 2, (Bs, 0, 2), color="firebrick", thickness=2, legend_label=r"$R_5 = -\phi^2 B^2/2$, $\phi = 1$")
right += point([(b, -b^2 / 2) for b in (0.5, 1, 1.5, 2)], size=40, color="royalblue", legend_label="SageManifolds values")
right.axes_labels(["$B$", "$R_5$"]); right.set_legend_options(loc="lower left", font_size=8)
fig = graphics_array([left, right]).matplotlib(figsize=[9, 3.8]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))
