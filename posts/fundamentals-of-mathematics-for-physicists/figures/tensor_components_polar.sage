# Figure: one tensor field, two charts. The type (0, 2) tensor field T = dx (x) dx on the plane has the constant
# components diag(1, 0) in Cartesian coordinates; in polar coordinates its components T_rr = cos^2(phi),
# T_rphi = -r sin(phi) cos(phi), T_phiphi = r^2 sin^2(phi) vary from point to point, although the tensor does not.
# The three panels show the polar components over the plane, computed with SageManifolds' change of frame.
# Output: figures/tensor_components_polar.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tensor_components_polar.svg")
M = Manifold(2, "R^2")
cart = M.chart("x y"); x, y = cart[:]
U = M.open_subset("U")
pol = U.chart(r"r:(0,+oo) ph:(0,2*pi):\varphi"); r, ph = pol[:]
pol_to_cart = pol.transition_map(cart.restrict(U), (r * cos(ph), r * sin(ph)))
pol_to_cart.set_inverse(sqrt(x**2 + y**2), atan2(y, x), check=False)
dx = cart.coframe()[0]
T = M.tensor_field(0, 2, name="T"); T[cart.frame(), :] = [[1, 0], [0, 0]]       # T = dx (x) dx
TU = T.restrict(U)
TU.display(pol.frame(), pol)                                   # triggers the change of frame
comps = TU.comp(pol.frame(), pol)
T_rr, T_rp, T_pp = comps[0, 0].expr(), comps[0, 1].expr(), comps[1, 1].expr()
xx, yy = var("xx yy", domain="real")
subs = {r: sqrt(xx**2 + yy**2), ph: atan2(yy, xx)}
panels = []
for expr, title in [(T_rr, r"$T_{rr}=\cos^2\varphi$"), (T_rp, r"$T_{r\varphi}=-r\sin\varphi\cos\varphi$"), (T_pp, r"$T_{\varphi\varphi}=r^2\sin^2\varphi$")]:
    f = expr.subs(subs)
    p = density_plot(f, (xx, -2, 2), (yy, -2, 2), cmap="RdBu", plot_points=80)
    p += text(title, (0, 2.35), fontsize=10, color="black")
    p.axes(False)
    panels.append(p)
g = graphics_array(panels)
fig = g.matplotlib(figsize=[9.5, 3.4]); fig.savefig(OUT, bbox_inches="tight")
print("[PASS] polar components of dx (x) dx are cos^2, -r sin cos, r^2 sin^2:", bool((T_rr - cos(ph)**2).simplify_full() == 0 and (T_rp + r*sin(ph)*cos(ph)).simplify_full() == 0 and (T_pp - r**2*sin(ph)**2).simplify_full() == 0))
print("[PASS] the components are symmetric, T_rphi = T_phir:", bool((comps[0, 1].expr() - comps[1, 0].expr()).simplify_full() == 0))
print("wrote", os.path.relpath(OUT))
