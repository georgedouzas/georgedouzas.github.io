# Figure: the dilaton-dependent couplings of the reduced theory. For a constant circle size phi (radius R phi in the
# conventions y ~ y + 2 pi R) the effective Newton constant of the Jordan frame is G/phi, the Einstein-frame Maxwell
# term -(1/4) phi^3 F^2 gives the gauge coupling e_eff = e phi^{-3/2}, and the Kaluza-Klein masses are |n|/(R phi^{3/2})
# in the Einstein frame (|n|/(R phi) in the Jordan frame); the charge-to-mass ratio q_n/m_n = sqrt(2) kappa does not
# depend on phi. The exponents are checked symbolically: the Weyl rescaling g = phi^{-1} gE of
# sqrt(-g) phi phi^2 g^{..} g^{..} F F gives phi^3 sqrt(-gE) gE^{..} gE^{..} F F, and the Klein-Gordon equation of the
# mode on phi^{-1} eta - phi^2 dy^2 gives the Einstein-frame mass. Output: figures/kkr_couplings.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kkr_couplings.svg")
p, n, R, kappa = var("p n R kappa"); assume(p > 0, R > 0, kappa > 0, n > 0)
# Weyl weight of the Maxwell term: sqrt(-g) = p^{-2} sqrt(-gE), g^{mu nu} = p gE^{mu nu} in four dimensions
weight = p^(-2) * p * p^2 * p * p
print("[PASS] Einstein-frame Maxwell weight: sqrt(-g) phi (phi^2 F^2)_g = phi^3 sqrt(-gE) (F^2)_E:", bool(weight == p^3))
M = Manifold(5, "M", structure="Lorentzian"); X = M.chart("t x v z y"); t, x, v, z, y = X[:]
G = M.metric("G"); G[0, 0] = 1 / p
for i in (1, 2, 3): G[i, i] = -1 / p
G[4, 4] = -p^2
f = function("f")(t); nab = G.connection()
red = (nab(nab(M.scalar_field(f * exp(I * n * y / R))).up(G)).trace().expr() / exp(I * n * y / R)).simplify_full()
mE2 = n^2 / (R^2 * p^3)
print("[PASS] Einstein-frame Kaluza-Klein mass squared n^2/(R^2 phi^3) from Box_5 on phi^{-1} eta - phi^2 dy^2:", bool((red - p * (diff(f, t, 2) + mE2 * f)).simplify_full() == 0))
e = sqrt(2) * kappa / R
q_eff = n * e * p^(-3 / 2); m_E = n / (R * p^(3 / 2))
print("[PASS] the charge-to-mass ratio q_n/m_n = n e phi^{-3/2}/(n/(R phi^{3/2})) = sqrt(2) kappa for every phi:", bool((q_eff / m_E - sqrt(2) * kappa).simplify_full() == 0))
s = var("s")
P = plot(1 / s, (s, 0.3, 3), color="royalblue", thickness=2, legend_label=r"$G_{\mathrm{eff}}/G = \phi^{-1}$ (Jordan frame)")
P += plot(s^(-3 / 2), (s, 0.3, 3), color="firebrick", thickness=2, legend_label=r"$e_{\mathrm{eff}}/e = m_1R = \phi^{-3/2}$ (Einstein frame)")
P += plot(s^(-1), (s, 0.3, 3), color="firebrick", thickness=1.2, linestyle="--", legend_label=r"$m_1 R = \phi^{-1}$ (Jordan frame)")
P += plot(1 + 0 * s, (s, 0.3, 3), color="darkgreen", thickness=2, linestyle=":", legend_label=r"$q_n/(\sqrt{2}\kappa\, m_n) = 1$")
P.axes_labels([r"$\phi$", ""]); P.set_legend_options(loc="upper right", font_size=8)
P.save(OUT, figsize=[7, 4], ymin=0, ymax=4)
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the same curves with hover, and the first levels of the Einstein-frame tower
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import plotly_html
_ps = [0.3 + 2.7 * k / 300 for k in range(301)]
_traces = [{"x": _ps, "y": [1 / q for q in _ps], "mode": "lines", "name": "G_eff/G = 1/phi (Jordan)", "line": {"color": "royalblue", "width": 3}},
           {"x": _ps, "y": [q ** -1.5 for q in _ps], "mode": "lines", "name": "e_eff/e = m_1 R = phi^(-3/2) (Einstein)", "line": {"color": "firebrick", "width": 3}},
           {"x": _ps, "y": [1 / q for q in _ps], "mode": "lines", "name": "m_1 R = 1/phi (Jordan)", "line": {"color": "firebrick", "width": 1.5, "dash": "dash"}},
           {"x": _ps, "y": [2 * q ** -1.5 for q in _ps], "mode": "lines", "name": "m_2 R (Einstein)", "line": {"color": "darkorange", "width": 1.5}, "visible": "legendonly"},
           {"x": _ps, "y": [1 for q in _ps], "mode": "lines", "name": "q_n/(sqrt2 kappa m_n) = 1", "line": {"color": "darkgreen", "width": 2, "dash": "dot"}}]
plotly_html("kkr_couplings", _traces, {"title": "Couplings and masses as functions of the circle size phi", "xaxis": {"title": "phi"}, "yaxis": {"title": "relative value", "range": [0, 4]}, "legend": {"orientation": "h", "y": -0.25}})
print("wrote figures/interactive/kkr_couplings.html")
