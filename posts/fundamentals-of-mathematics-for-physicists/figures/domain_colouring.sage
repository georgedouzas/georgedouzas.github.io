# Figure: holomorphic functions seen in two ways. Left: domain colouring of f(z) = (z - 1)/(z^2 + 1), hue = arg f,
# brightness = |f|; the zero at z = 1 and the poles at z = +-i are the points where all colours meet. Right: the
# image of the Cartesian grid under z -> z^2, a conformal map: the image curves still meet at right angles.
# Output: figures/domain_colouring.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "domain_colouring.svg")
z = var("z")
f = (z - 1)/(z^2 + 1)
p = complex_plot(f, (-2.2, 2.2), (-2.2, 2.2), plot_points=220)
p += point((1, 0), size=40, color="white") + text("zero", (1.0, 0.25), fontsize=9, color="white")
for s in (1, -1):
    p += point((0, s), size=40, color="black") + text("pole", (0.35, s), fontsize=9, color="black")
p.set_aspect_ratio(1)
q = Graphics()
for c in [k/4 for k in range(-6, 7)]:
    q += parametric_plot(((c^2 - var("t")^2), 2*c*var("t")), (var("t"), -1.5, 1.5), color="royalblue", thickness=0.8)   # image of Re z = c
    q += parametric_plot(((var("t")^2 - c^2), 2*var("t")*c), (var("t"), -1.5, 1.5), color="firebrick", thickness=0.8)  # image of Im z = c
q += text("z -> z^2: images of Re z = c (blue)", (0, -4.4), fontsize=9, color="royalblue") + text("and of Im z = c (red) meet at right angles", (0, -4.9), fontsize=9, color="firebrick")
q.set_aspect_ratio(1); q.axes(False)
graphics_array([[p, q]]).matplotlib(figsize=(10.5, 4.8)).savefig(OUT, bbox_inches="tight")
# checks for prp-cauchy-riemann-examples
x, y = var("x y", domain="real")
def parts(expr):
    e = expr.subs(z=x + I*y).expand()
    return e.real_part().simplify_full(), e.imag_part().simplify_full()
def cr(u, v):
    return bool((diff(u, x) - diff(v, y)).simplify_full() == 0) and bool((diff(u, y) + diff(v, x)).simplify_full() == 0)
u2, v2 = parts(z^2); u3, v3 = parts(z^3); ue, ve = parts(exp(z))
print("[PASS] z^2 = (x^2 - y^2) + i 2xy satisfies the Cauchy-Riemann equations:", bool(u2 == x^2 - y^2) and bool(v2 == 2*x*y) and cr(u2, v2))
print("[PASS] z^3 and e^z = e^x cos y + i e^x sin y satisfy the Cauchy-Riemann equations:", cr(u3, v3) and cr(ue, ve))
print("[PASS] conj(z) = x - i y violates the Cauchy-Riemann equations:", not cr(x, -y))
print("[PASS] u = x^2 - y^2 and u = e^x cos y are harmonic (Laplacian zero):", bool((diff(u2, x, 2) + diff(u2, y, 2)).simplify_full() == 0) and bool((diff(ue, x, 2) + diff(ue, y, 2)).simplify_full() == 0))
J = jacobian([u2, v2], [x, y])
print("[PASS] the Jacobian of z -> z^2 has determinant |2z|^2 = 4(x^2 + y^2) and orthogonal columns of equal length (a rotation times a scaling):", bool((J.det() - 4*(x^2 + y^2)).simplify_full() == 0) and bool((J.column(0)*J.column(1)).simplify_full() == 0) and bool((J.column(0)*J.column(0) - J.column(1)*J.column(1)).simplify_full() == 0))
print("[PASS] f = (z - 1)/(z^2 + 1) has its zero at z = 1 and its poles at z = +-i:", [r for r, m in (z - 1).roots(ring=QQbar)] == [1] and sorted(str(r) for r, m in (z^2 + 1).roots(ring=QQbar)) == ["-1*I", "1*I"])
print("wrote", os.path.relpath(OUT))

# interactive version (design D21): the grid and its image under z^2, hover shows the point and its image
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _interactive import plotly_html
_traces = []
_cs = [k/4 for k in range(-6, 7)]
_ts = [-1.5 + 3*j/60 for j in range(61)]
for _k, _c in enumerate(_cs):
    _pts = [complex(_c, _t)**2 for _t in _ts]
    _traces.append({"x": [w.real for w in _pts], "y": [w.imag for w in _pts], "mode": "lines", "line": {"color": "royalblue", "width": 1},
                    "name": "image of Re z = c", "legendgroup": "re", "showlegend": _k == 0,
                    "customdata": [[_c, _t] for _t in _ts], "hovertemplate": "z = %{customdata[0]:.2f} + %{customdata[1]:.2f} i<br>z^2 = %{x:.2f} + %{y:.2f} i<extra></extra>"})
    _pts = [complex(_t, _c)**2 for _t in _ts]
    _traces.append({"x": [w.real for w in _pts], "y": [w.imag for w in _pts], "mode": "lines", "line": {"color": "firebrick", "width": 1},
                    "name": "image of Im z = c", "legendgroup": "im", "showlegend": _k == 0,
                    "customdata": [[_t, _c] for _t in _ts], "hovertemplate": "z = %{customdata[0]:.2f} + %{customdata[1]:.2f} i<br>z^2 = %{x:.2f} + %{y:.2f} i<extra></extra>"})
plotly_html("domain_colouring", _traces, {"title": "The Cartesian grid under z -> z^2 (hover for a point and its image)",
            "xaxis": {"scaleanchor": "y", "title": "Re z^2"}, "yaxis": {"title": "Im z^2"}, "legend": {"orientation": "h"}})
print("wrote figures/interactive/domain_colouring.html")
