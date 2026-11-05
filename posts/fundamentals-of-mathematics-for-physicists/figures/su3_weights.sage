# Figure: the root diagram of su(3) and the weight diagrams of its fundamental (3) and adjoint (8) representations,
# in the (lambda_3, lambda_8) eigenvalue plane (the Cartan subalgebra). Roots are the weights of the adjoint:
# the eigenvalue pairs of (ad lambda_3, ad lambda_8) on lambda_1 +- i lambda_2, lambda_4 +- i lambda_5,
# lambda_6 +- i lambda_7, plus the two zero weights. Output: figures/su3_weights.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "su3_weights.svg")
lam = [matrix(SR, [[0,1,0],[1,0,0],[0,0,0]]), matrix(SR, [[0,-I,0],[I,0,0],[0,0,0]]), matrix(SR, [[1,0,0],[0,-1,0],[0,0,0]]),
       matrix(SR, [[0,0,1],[0,0,0],[1,0,0]]), matrix(SR, [[0,0,-I],[0,0,0],[I,0,0]]), matrix(SR, [[0,0,0],[0,0,1],[0,1,0]]),
       matrix(SR, [[0,0,0],[0,0,-I],[0,I,0]]), matrix(SR, [[1,0,0],[0,1,0],[0,0,-2]])/sqrt(3)]
H3, H8 = lam[2], lam[7]
def charge(X, Hc):                                        # [H, X] = q X  ->  q
    C = Hc*X - X*Hc
    for e in zip(C.list(), X.list()):
        if e[1] != 0: return (e[0]/e[1]).simplify_full()
roots = []
for a, b in [(0, 1), (3, 4), (5, 6)]:
    for s in (1, -1):
        E = lam[a] + s*I*lam[b]
        roots.append((charge(E, H3), charge(E, H8)))
# fundamental weights: the diagonal entries of (lambda_3, lambda_8)
fund = [(H3[i, i], H8[i, i]) for i in range(3)]
p = Graphics()
for r in roots:
    p += arrow((0, 0), r, color="firebrick", width=1.5)
p += point(roots + [(0, 0)], size=55, color="firebrick", zorder=5)
p += point(fund, size=55, color="royalblue", zorder=6, marker="s")
for w, lab in zip(fund, ["u", "d", "s"]):
    p += text(lab, (w[0] + 0.12, w[1] + 0.12), color="royalblue", fontsize=11)
p += text("roots (weights of 8)", (0, -1.95), color="firebrick", fontsize=11) + text("weights of 3", (0, -2.2), color="royalblue", fontsize=11)
p.set_aspect_ratio(1); p.axes_labels([r"$\lambda_3$", r"$\lambda_8$"])
p.save(OUT, figsize=5)
print("[PASS] roots:", sorted([(str(r[0]), str(r[1])) for r in roots]))
print("[PASS] the roots are the pairwise differences of the fundamental weights:",
      set((str((a[0]-b[0]).simplify_full()), str((a[1]-b[1]).simplify_full())) for a in fund for b in fund if a != b) == set((str(r[0]), str(r[1])) for r in roots))
print("[PASS] all roots have the same length sqrt(4) = 2:", all((r[0]^2 + r[1]^2).simplify_full() == 4 for r in roots))
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the weight diagrams of 3, 3bar, 8 and 10 from Sage's WeylCharacterRing, in the same
# (lambda_3, lambda_8) plane; each representation is a legend entry that can be toggled, and hovering a weight shows its
# coordinates, its multiplicity and its lambda_8-charge row (the su(2) x u(1) branching of the section on branching)
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import plotly_html
_A2 = WeylCharacterRing("A2", style="coroots")
def _plane(w):                                   # ambient weight (w1, w2, w3) -> (lambda_3, lambda_8) eigenvalues
    return (w[0] - w[1], (w[0] + w[1] - 2 * w[2]) / sqrt(3))
_reps = [("3", _A2(1, 0), "royalblue", "square", 16), ("3bar", _A2(0, 1), "seagreen", "diamond", 16),
         ("8 (adjoint)", _A2(1, 1), "firebrick", "circle", 13), ("10", _A2(3, 0), "darkorange", "triangle-up", 11)]
_traces = []
for _r in roots:
    _traces.append({"x": [0, float(_r[0])], "y": [0, float(_r[1])], "mode": "lines", "line": {"color": "firebrick", "width": 1}, "opacity": 0.4, "hoverinfo": "skip", "showlegend": False})
_check_roots = None
for _name, _chi, _col, _sym, _size in _reps:
    _wm = _chi.weight_multiplicities()
    _pts = [(_plane(w.to_vector()), m) for w, m in _wm.items()]
    if _name.startswith("8"):
        _check_roots = sorted((str(SR(p[0]).simplify_full()), str(SR(p[1]).simplify_full())) for p, m in _pts if p != (0, 0))
    _traces.append({"x": [float(p[0]) for p, m in _pts], "y": [float(p[1]) for p, m in _pts], "mode": "markers", "name": "%s (dim %d)" % (_name, _chi.degree()),
                    "marker": {"color": _col, "symbol": _sym, "size": [_size + 6 * (m - 1) for p, m in _pts], "opacity": 0.8, "line": {"color": "white", "width": 1}},
                    "text": ["%s: weight (%s, %s), multiplicity %d" % (_name, SR(p[0]).simplify_full(), SR(p[1]).simplify_full(), m) for p, m in _pts], "hoverinfo": "text"})
plotly_html("su3_weights", _traces, {"title": "Weights of su(3) representations (click the legend to toggle; hover a weight)",
            "xaxis": {"title": "lambda_3 eigenvalue", "scaleanchor": "y", "zeroline": True}, "yaxis": {"title": "lambda_8 eigenvalue", "zeroline": True},
            "legend": {"orientation": "h"}})
print("[PASS] the nonzero weights of the adjoint from the WeylCharacterRing are the six roots of the Gell-Mann computation:",
      _check_roots == sorted((str(r[0]), str(r[1])) for r in roots))
print("[PASS] the 10 has ten weights of multiplicity one:", len(_A2(3, 0).weight_multiplicities()) == 10 and set(_A2(3, 0).weight_multiplicities().values()) == {1})
print("wrote figures/interactive/su3_weights.html")
