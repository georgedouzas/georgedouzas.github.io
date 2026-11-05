# Figure: Weyl rescaling. Left: the unit sphere rescaled by Omega(theta) = 1 + cos(theta)/2, a two-dimensional picture
# of a conformal factor (the surface drawn with radius Omega). Right: the curvature scalar before and after the
# rescaling, R~ = Omega^{-2}(R - 2 Box ln Omega) in two dimensions, checked with SageManifolds; the four-dimensional
# formula R~ = Omega^{-2}(R - 6 Omega^{-1} Box Omega) is checked on de Sitter space with Omega = Omega(t), in the
# mostly-minus signature (+,-,-,-) of the tutorial (the formula is signature independent: R and Box flip together).
# Output: figures/kk_einstein_frame.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kk_einstein_frame.svg")
S = Manifold(2, "S"); X = S.chart("th ph"); th, ph = X[:]
Om = 1 + cos(th) / 2
g = S.metric("g"); g[0, 0] = 1; g[1, 1] = sin(th)^2
gt = S.metric("gt"); gt[0, 0] = Om^2; gt[1, 1] = Om^2 * sin(th)^2
R0 = g.ricci_scalar().expr().simplify_full(); Rt = gt.ricci_scalar().expr().simplify_full()
nab = g.connection(); lnO = S.scalar_field(log(Om))
boxlnO = (nab(nab(lnO).up(g)).trace()).expr().simplify_full()           # Box ln Omega on the unit sphere
print("R =", R0, "; R~ =", Rt)
print("[PASS] two dimensions: R~ = Omega^{-2}(R - 2 Box ln Omega) for Omega = 1 + cos(theta)/2:", bool((Rt - (R0 - 2 * boxlnO) / Om^2).simplify_full() == 0))
# four dimensions: de Sitter in flat slicing, Omega = Omega(t)
M = Manifold(4, "M", structure="Lorentzian"); Y = M.chart("t x y z"); t, x, y, z = Y[:]
H = var("H"); Omega = function("Omega")(t)
d = M.metric("d"); d[0, 0] = 1
for i in (1, 2, 3): d[i, i] = -exp(2 * H * t)
dt = M.metric("dt"); dt[0, 0] = Omega^2
for i in (1, 2, 3): dt[i, i] = -Omega^2 * exp(2 * H * t)
R4 = d.ricci_scalar().expr(); R4t = dt.ricci_scalar().expr()
nab4 = d.connection(); Os = M.scalar_field(Omega)
boxO = (nab4(nab4(Os).up(d)).trace()).expr()
print("[PASS] four dimensions: R~ = Omega^{-2} R - 6 Omega^{-3} Box Omega on de Sitter space with Omega = Omega(t):", bool((R4t - (R4 / Omega^2 - 6 * boxO / Omega^3)).simplify_full() == 0))
tt = var("tt")
left = parametric_plot((Om.subs(th=tt) * sin(tt), Om.subs(th=tt) * cos(tt)), (tt, 0, pi), color="firebrick", thickness=2, legend_label=r"$\Omega(\theta)$ profile")
left += parametric_plot((sin(tt), cos(tt)), (tt, 0, pi), color="royalblue", linestyle="--", legend_label="unit sphere")
left.set_aspect_ratio(1); left.axes(False); left.set_legend_options(loc="lower left", font_size=8)
right = plot(R0, (tt, 0.05, pi - 0.05), color="royalblue", linestyle="--", thickness=2, legend_label="$R = 2$")
right += plot(Rt.subs(th=tt), (tt, 0.05, pi - 0.05), color="firebrick", thickness=2, legend_label=r"$\tilde R = \Omega^{-2}(R - 2\nabla^2\ln\Omega)$")
right.axes_labels([r"$\theta$", "curvature"]); right.set_legend_options(loc="upper left", font_size=8)
fig = graphics_array([left, right]).matplotlib(figsize=[9, 3.8]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))
