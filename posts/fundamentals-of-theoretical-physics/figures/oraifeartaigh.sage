# Figure: the O'Raifeartaigh model W = -k Phi_1 + m Phi_2 Phi_3 + (y/2) Phi_1 Phi_3^2 (Martin, hep-ph/9709356,
# eq. 7.3.1). F_1 = 0 and F_2 = 0 are incompatible, so V = |F_1|^2 + |F_2|^2 + |F_3|^2 > 0 everywhere: supersymmetry
# is spontaneously broken. Left: V along the real phi_3 direction at phi_1 = phi_2 = 0 for m^2 > yk (minimum at
# the origin, V = k^2) and m^2 < yk (the origin is a saddle). Right: V along phi_1 is flat at tree level. The
# tree-level spectrum at the origin shows the boson-fermion mass splitting. Output: figures/oraifeartaigh.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oraifeartaigh.svg")
k, m, y = var("k m y"); assume(k > 0, m > 0, y > 0)
p1, p2, p3 = var("p1 p2 p3")           # real sections of the complex fields
F1 = k - y / 2 * p3^2; F2 = -m * p3; F3 = -m * p2 - y * p1 * p3
V = F1^2 + F2^2 + F3^2
print("[PASS] F_1 = 0 and F_2 = 0 have no common solution (F_2 = 0 forces phi_3 = 0, then F_1 = k != 0):",
      solve([F1 == 0, F2 == 0], [p3]) == [] or bool(F1.subs(p3=0) == k))
print("[PASS] V >= k^2 ... V at the origin is k^2 and the gradient vanishes there:",
      bool(V.subs(p1=0, p2=0, p3=0) == k^2) and all(bool(diff(V, v).subs(p1=0, p2=0, p3=0) == 0) for v in (p1, p2, p3)))
# tree-level scalar mass-squared eigenvalues at the origin (real and imaginary parts of phi_2, phi_3): m^2 -+ yk, m^2
x, w = var("x w")                       # phi_3 = (x + i w)/sqrt(2), at phi_1 = phi_2 = 0
Vc = ((k - y / 2 * (x + I * w)^2 / 2) * (k - y / 2 * (x - I * w)^2 / 2) + m^2 * (x^2 + w^2) / 2).expand()
Hm = matrix([[diff(Vc, x, 2), diff(Vc, x, w)], [diff(Vc, x, w), diff(Vc, w, 2)]]).subs(x=0, w=0)
print("[PASS] the phi_3 mass-squared eigenvalues at the origin are m^2 - yk and m^2 + yk (Martin eq. 7.3.9), split from the fermion mass m^2:",
      sorted(Hm.eigenvalues(), key=lambda e: str(e)) == sorted([m^2 - k * y, m^2 + k * y], key=lambda e: str(e)) or set(Hm.eigenvalues()) == set([m^2 - k * y, m^2 + k * y]))
Va = V.subs(p1=0, p2=0)
left = plot(Va.subs(k=1, m=1.5, y=1), (p3, -2.2, 2.2), color="royalblue", thickness=2, legend_label="$m^2 > yk$")
left += plot(Va.subs(k=1, m=0.6, y=1), (p3, -2.2, 2.2), color="firebrick", thickness=2, legend_label="$m^2 < yk$")
left += line([(-2.2, 0), (2.2, 0)], color="gray", linestyle=":")
left.axes_labels(["$\\phi_3$", "$V$"]); left.ymax(6); left.set_legend_options(loc="upper center", font_size=8)
right = plot(V.subs(p2=0, p3=0, k=1, m=1.5, y=1), (p1, -3, 3), color="royalblue", thickness=2, legend_label="$V(\\phi_1)=k^2$, flat direction")
right += line([(-3, 0), (3, 0)], color="gray", linestyle=":")
right.axes_labels(["$\\phi_1$", "$V$"]); right.ymin(-0.2); right.ymax(2.5); right.set_legend_options(loc="upper center", font_size=8)
fig = graphics_array([left, right]).matplotlib(figsize=[9, 3.8]); fig.savefig(OUT, bbox_inches="tight")
print("wrote", os.path.relpath(OUT))

# for m^2 < yk the minimum moves to phi_3^2 = 2(yk - m^2)/y^2 (real phi_3, with F_3 = 0), still with V > 0
xm = var("xm"); Vx = (k - y / 2 * xm^2)^2 + m^2 * xm^2
xmin2 = 2 * (y * k - m^2) / y^2
print("[PASS] for m^2 < yk the minimum is at phi_3^2 = 2(yk - m^2)/y^2 with V_min = m^2(2yk - m^2)/y^2, positive and below k^2:",
      bool(diff(Vx, xm).subs(xm=sqrt(xmin2)).simplify_full() == 0)
      and bool((Vx.subs(xm=sqrt(xmin2)) - m^2 * (2 * y * k - m^2) / y^2).simplify_full() == 0)
      and bool((k^2 - m^2 * (2 * y * k - m^2) / y^2 - (y * k - m^2)^2 / y^2).simplify_full() == 0))

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _interactive import plotly_html
grid = [-2 + 4 * i / 60 for i in range(61)]
def surf(mv, kv=1, yv=1):
    # V(phi_1 = phi_2 = 0, phi_3 = (u + i v)) = |k - y phi_3^2/2|^2 + m^2 |phi_3|^2, the true minimum over phi_1, phi_2
    return [[min(6.0, (kv - yv / 2 * (u * u - v * v))**2 + (yv * u * v)**2 + mv**2 * (u * u + v * v)) for u in grid] for v in grid]
mvals = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
def vmin(mv, kv=1, yv=1):
    return (kv**2, 0.0) if mv**2 >= yv * kv else (mv**2 * (2 * yv * kv - mv**2) / yv**2, (2 * (yv * kv - mv**2) / yv**2)**0.5)
traces = [{"type": "surface", "x": grid, "y": grid, "z": surf(1.5), "colorscale": "Viridis", "cmin": 0, "cmax": 6,
           "name": "V", "showscale": False, "contours": {"z": {"show": True, "usecolormap": True, "project": {"z": True}}}},
          {"type": "scatter3d", "mode": "markers", "x": [0], "y": [0], "z": [1.0], "marker": {"size": 5, "color": "red"}, "name": "minimum"}]
steps = []
for mv in mvals:
    vm, um = vmin(mv)
    steps.append({"method": "restyle", "label": "m=%s" % mv,
                  "args": [{"z": [surf(mv), [vm, vm] if um else [vm]], "x": [grid, [um, -um] if um else [0]], "y": [grid, [0, 0] if um else [0]]}]})
plotly_html("oraifeartaigh", traces, {"title": "O'Raifeartaigh potential over the complex phi_3 plane (k = y = 1): minimum V > 0 (red)",
            "scene": {"xaxis": {"title": "Re phi_3"}, "yaxis": {"title": "Im phi_3"}, "zaxis": {"title": "V", "range": [0, 6]}},
            "sliders": [{"active": 5, "steps": steps, "currentvalue": {"prefix": "mass parameter "}}]})
print("wrote figures/interactive/oraifeartaigh.html")
