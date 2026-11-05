# Figure: stationary action for a field. For a static scalar field on the interval [0, 1] with phi(0) = 0, phi(1) = 1,
# the energy functional E[phi] = int (phi'^2 + m^2 phi^2)/2 dx is stationary at the solution phi'' = m^2 phi,
# phi = sinh(m x)/sinh(m) (black, left), and for the varied fields phi + eps sin(pi x) (blue) E(eps) is a parabola
# with its minimum at eps = 0 (right). Computed symbolically. Output: figures/field_action.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "field_action.svg")
x, eps = var("x eps", domain="real")
m = 2
phicl = sinh(m * x) / sinh(m)
eta = sin(pi * x)
phi = phicl + eps * eta
E = integrate((diff(phi, x)**2 + m**2 * phi**2) / 2, x, 0, 1).simplify_full()
left = plot(phicl, (x, 0, 1), color="black", thickness=2.5, legend_label=r"solution $\sinh(mx)/\sinh m$")
for e in [-0.4, -0.2, 0.2, 0.4]:
    left += plot(phi.subs(eps=e), (x, 0, 1), color="royalblue", alpha=0.6, thickness=1)
left += text(r"varied fields $\phi+\varepsilon\sin\pi x$ (blue)", (0.4, 1.05), fontsize=9, color="royalblue")
left.axes_labels([r"$x$", r"$\phi$"]); left.set_legend_options(loc="upper left", font_size=8)
right = plot(E, (eps, -0.6, 0.6), color="firebrick", thickness=2)
right += point((0, E.subs(eps=0)), color="black", size=40) + text(r"$E(\varepsilon)$ stationary at $\varepsilon=0$", (0, E.subs(eps=0.6) * 0.97), fontsize=9)
right.axes_labels([r"$\varepsilon$", r"$E$"])
g = graphics_array([left, right]); fig = g.matplotlib(figsize=[9, 3.8]); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] the solution satisfies phi'' = m^2 phi with the boundary values 0 and 1:", bool((diff(phicl, x, 2) - m**2 * phicl).simplify_full() == 0) and bool(phicl.subs(x=0) == 0) and bool(phicl.subs(x=1) == 1))
print("[PASS] E(eps) has no linear term: dE/deps = 0 at eps = 0:", bool(diff(E, eps).subs(eps=0).simplify_full() == 0))
print("[PASS] E(eps) - E(0) = eps^2 (pi^2 + m^2)/4 > 0 for eps != 0, so the solution is a minimum:", bool((E - E.subs(eps=0) - eps**2 * (pi**2 + m**2) / 4).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
# interactive version: a slider moves along the one-parameter family phi + eps sin(pi x) and the marker along E(eps)
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
xs = [k / 100 for k in range(101)]; es = [round(-0.6 + 0.05 * k, 2) for k in range(25)]
fcl = fast_callable(phicl, vars=[x]); fE = fast_callable(E, vars=[eps]); fet = fast_callable(eta, vars=[x])
_traces = [{"x": xs, "y": [fcl(a) for a in xs], "mode": "lines", "name": "solution", "line": {"color": "black", "width": 3}},
           {"x": [e for e in [k / 100 - 0.6 for k in range(121)]], "y": [fE(k / 100 - 0.6) for k in range(121)], "mode": "lines", "name": "E(eps)", "line": {"color": "firebrick", "width": 2}, "xaxis": "x2", "yaxis": "y2"}]
for ev in es:
    _traces.append({"x": xs, "y": [fcl(a) + ev * fet(a) for a in xs], "mode": "lines", "name": "varied field", "line": {"color": "royalblue", "width": 2}, "visible": abs(ev - 0.3) < 1e-9})
    _traces.append({"x": [ev], "y": [fE(ev)], "mode": "markers", "name": "E of the varied field", "marker": {"color": "royalblue", "size": 11}, "xaxis": "x2", "yaxis": "y2", "visible": abs(ev - 0.3) < 1e-9})
steps = []
for i, ev in enumerate(es):
    vis = [True, True] + [False] * (2 * len(es)); vis[2 + 2 * i] = vis[3 + 2 * i] = True
    steps.append({"method": "update", "label": "%.2f" % ev, "args": [{"visible": vis}, {"title": "eps = %.2f:  E(eps) = %.4f,  E(0) = %.4f" % (ev, fE(ev), fE(0))}]})
plotly_html("field_action", _traces, {"title": "eps = 0.30:  E(eps) = %.4f,  E(0) = %.4f" % (fE(0.3), fE(0)), "showlegend": False,
            "xaxis": {"title": "x", "domain": [0, 0.46]}, "yaxis": {"title": "phi", "range": [-0.7, 1.3]},
            "xaxis2": {"title": "eps", "domain": [0.56, 1], "anchor": "y2"}, "yaxis2": {"title": "E", "anchor": "x2"},
            "sliders": [{"active": 18, "currentvalue": {"prefix": "eps = "}, "pad": {"t": 40}, "steps": steps}]})
print("wrote figures/interactive/field_action.html")
