# Figure: the orbifold T^2/Z_3. The hexagonal lattice Lambda = Z + Z omega (omega = e^{2 pi i/3}), a cell of the torus
# T^2 = C/Lambda (the two equilateral triangles), the three fixed points of the rotation z -> omega z (the lattice
# points, mod Lambda, and the centres of the two triangles) and a fundamental domain of the Z_3 action (the rhombus
# with corners 0, the two centres and 1 + omega, one third of the cell). Checks: the fixed points of z -> omega z on
# T^2 by exhaustive search over (1/3)Lambda and by the Lefschetz number det(1 - M) = 3; 27 fixed points on
# T^6 = T^2 x T^2 x T^2; the Z_3 action z^i -> omega z^i preserves dz^1 dz^2 dz^3 (Z_3 in SU(3)); the untwisted
# Hodge numbers h^{1,1} = 9, h^{2,1} = 0 and, with one class per fixed point, h^{1,1} = 36; the orbifold Euler number
# (1/|G|) sum over commuting pairs = 72 = 2(36 - 0); the area of the fundamental domain is 1/3 of the cell.
# Output: figures/z3_orbifold.svg, interactive figures/interactive/z3_orbifold.html.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "z3_orbifold.svg")
K.<w> = CyclotomicField(3)                  # w = omega, w^2 + w + 1 = 0
M = matrix(ZZ, [[0, -1], [1, -1]])          # z = x + y w  ->  w z = -y + (x - y) w in lattice coordinates
print("[PASS] in lattice coordinates z = x + y omega the rotation is M = [[0, -1], [1, -1]], of order 3:",
      all((w * (a + b * w)) == (M * vector([a, b]))[0] + (M * vector([a, b]))[1] * w for a in range(-2, 3) for b in range(-2, 3)) and M ** 3 == 1)
# fixed points mod Lambda: (M - 1) v in Z^2; they lie in (1/3) Z^2 since det(M - 1) = 3
cands = [vector(QQ, [a / 3, b / 3]) for a in range(3) for b in range(3)]
fixed = [v for v in cands if all(x in ZZ for x in (M - 1) * v)]
print("[PASS] z -> omega z has exactly 3 fixed points on T^2: (0,0), (2/3,1/3), (1/3,2/3) in lattice coordinates:",
      fixed == [vector(QQ, [0, 0]), vector(QQ, [1/3, 2/3]), vector(QQ, [2/3, 1/3])])
print("[PASS] Lefschetz number: det(1 - M) = 3 = number of fixed points:", (1 - M).det() == 3 == len(fixed))
print("[PASS] on T^6 = (T^2)^3 with z^i -> omega z^i: 3^3 = 27 fixed points:", len(fixed) ** 3 == 27)
print("[PASS] the action preserves dz^1 dz^2 dz^3 (omega^3 = 1), so Z_3 lies in SU(3) and one supersymmetry survives:", w ** 3 == 1)
# untwisted Hodge numbers: dz^i dzbar^j picks up omega * omegabar = 1, dz^i dz^j dzbar^k picks up omega^2 * omegabar = omega
wb = w.conjugate()
h11_untw = sum(1 for i in range(3) for j in range(3) if w * wb == 1)
h21_untw = sum(1 for i in range(3) for j in range(i + 1, 3) for k in range(3) if w * w * wb == 1)
print("[PASS] untwisted sector: h^{1,1} = 9 (all dz^i dzbar^j invariant), h^{2,1} = 0 (dz dz dzbar picks up omega):", h11_untw == 9 and h21_untw == 0)
h11 = h11_untw + 27
print("[PASS] twisted sectors add one (1,1)-class per fixed point: h^{1,1} = 9 + 27 = 36, h^{2,1} = 0:", h11 == 36)
# orbifold Euler number: (1/|G|) sum over commuting pairs (g, h) of chi(fixed set of both); chi(T^6) = 0
G = [0, 1, 2]
chi_orb = sum((0 if (a, b) == (0, 0) else 27) for a in G for b in G) / 3
print("[PASS] orbifold Euler number (1/3)(0 + 8 * 27) = 72 = 2(h^{1,1} - h^{2,1}):", chi_orb == 72 == 2 * (h11 - 0))
# geometry in the plane
om = vector(RDF, [-0.5, sqrt(3.0) / 2]); one = vector(RDF, [1.0, 0.0])
def P(x, y): return x * one + y * om
cell = [P(0, 0), P(1, 0), P(1, 1), P(0, 1)]
c1, c2 = P(2/3, 1/3), P(1/3, 2/3)
dom = [P(0, 0), c1, P(1, 1), c2]
def area(poly):
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))) / 2
print("[PASS] the rhombus (0, centre, 1 + omega, centre) has 1/3 of the area of the cell, a fundamental domain of Z_3:", abs(area(dom) - area(cell) / 3) < 1e-12)
print("[PASS] the two non-lattice fixed points are the centres of the equilateral triangles (0, 1, 1 + omega) and (0, 1 + omega, omega):",
      (c1 - (P(0, 0) + P(1, 0) + P(1, 1)) / 3).norm() < 1e-12 and (c2 - (P(0, 0) + P(1, 1) + P(0, 1)) / 3).norm() < 1e-12)

G2 = Graphics()
for a in range(-2, 4):
    for b in range(-2, 4):
        p = P(a, b)
        if -1.6 < p[0] < 2.1 and -1.0 < p[1] < 2.0:
            G2 += point(p, color="black", size=18)
G2 += polygon(cell, fill=False, color="grey", thickness=1.2, linestyle="--")
G2 += line([P(0, 0), P(1, 1)], color="grey", thickness=0.8, linestyle=":")
G2 += polygon(dom, color="lightsteelblue", alpha=0.7, edgecolor="royalblue", thickness=1.5)
G2 += point([P(0, 0), P(1, 1), P(1, 0), P(0, 1)], color="firebrick", size=60, marker="o")
G2 += point([c1], color="darkgreen", size=70, marker="^")
G2 += point([c2], color="darkorange", size=70, marker="s")
G2 += text("$0$", (-0.12, -0.12), color="firebrick", fontsize=12)
G2 += text("$1$", (1.08, -0.12), color="firebrick", fontsize=12)
G2 += text("$\\omega$", (-0.62, 0.92), color="firebrick", fontsize=12)
G2 += text("$1+\\omega$", (0.6, 1.0), color="firebrick", fontsize=12)
G2 += text("$\\frac{2+\\omega}{3}$", (0.72, 0.18), color="darkgreen", fontsize=12)
G2 += text("$\\frac{1+2\\omega}{3}$", (-0.22, 0.62), color="darkorange", fontsize=12)
G2.axes(False); G2.set_aspect_ratio(1)
fig = G2.matplotlib(figsize=[6.2, 5.2]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

# Interactive version: slide the rotation angle theta from 0 to 240 degrees. The fundamental domain (blue) is rotated
# about the origin; at theta = 120 and 240 degrees it lands, modulo the lattice, on the other two thirds of the cell,
# and the fixed points (markers) are mapped to themselves modulo the lattice. Hover for coordinates.
import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html, _clean
import json, math
def _rot(p, th): return (math.cos(th) * p[0] - math.sin(th) * p[1], math.sin(th) * p[0] + math.cos(th) * p[1])
_dom = [tuple(map(float, q)) for q in dom] + [tuple(map(float, dom[0]))]
def _data(deg):
    th = math.radians(deg); pts = [_rot(q, th) for q in _dom]
    return [{"x": [q[0] for q in pts], "y": [q[1] for q in pts]}]
_lat = [tuple(map(float, P(a, b))) for a in range(-2, 4) for b in range(-2, 4)]
_lat = [q for q in _lat if -1.7 < q[0] < 2.2 and -1.7 < q[1] < 2.0]
_cellc = [tuple(map(float, q)) for q in cell] + [tuple(map(float, cell[0]))]
_frames, _steps = [], []
for deg in range(0, 241, 10):
    _frames.append({"name": str(deg), "data": _data(deg), "traces": [2]})
    _steps.append({"label": str(deg), "method": "animate", "args": [[str(deg)], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]})
_traces = [{"x": [q[0] for q in _lat], "y": [q[1] for q in _lat], "mode": "markers", "name": "lattice (fixed point 0)", "marker": {"color": "firebrick", "size": 8}},
           {"x": [q[0] for q in _cellc], "y": [q[1] for q in _cellc], "mode": "lines", "name": "torus cell", "line": {"color": "grey", "dash": "dash"}},
           dict(_data(0)[0], mode="lines", fill="toself", name="Z_3 fundamental domain rotated by theta", line={"color": "royalblue"}, fillcolor="rgba(70,130,180,0.35)"),
           {"x": [float(c1[0])], "y": [float(c1[1])], "mode": "markers", "name": "fixed point (2 + omega)/3", "marker": {"color": "darkgreen", "size": 12, "symbol": "triangle-up"}},
           {"x": [float(c2[0])], "y": [float(c2[1])], "mode": "markers", "name": "fixed point (1 + 2 omega)/3", "marker": {"color": "darkorange", "size": 12, "symbol": "square"}}]
_layout = {"title": "T^2/Z_3: rotate the fundamental domain by theta (degrees); at 120 and 240 it tiles the cell mod the lattice",
           "xaxis": {"range": [-1.7, 2.2], "title": "Re z"}, "yaxis": {"range": [-1.7, 2.0], "scaleanchor": "x", "title": "Im z"},
           "sliders": [{"active": 0, "currentvalue": {"prefix": "theta = "}, "steps": _steps, "pad": {"t": 30}}], "legend": {"orientation": "h", "y": -0.3}}
_path = plotly_html("z3_orbifold", _traces, _layout)
_html = open(_path).read().replace("</script></body>", "Plotly.addFrames('fig', %s);</script></body>" % json.dumps(_clean(_frames)))
open(_path, "w").write(_html)
# at 120 degrees the rotated domain differs from a piece of the cell by a lattice vector: its corners are fixed points mod Lambda
_c120 = [_rot(q, 2 * math.pi / 3) for q in _dom[:4]]
_ok = all(any(abs(q[0] - (l[0] + f[0])) < 1e-9 and abs(q[1] - (l[1] + f[1])) < 1e-9 for l in _lat for f in [(0.0, 0.0), tuple(map(float, c1)), tuple(map(float, c2))]) for q in [_c120[0], _c120[1], _c120[3]])
print("[PASS] slider: the corners 0 and the two centres of the domain rotated by 120 degrees are fixed points modulo the lattice:", _ok)
print("wrote figures/interactive/z3_orbifold.html")
