# Figure: why partial derivatives of components are not a tensor. The constant vector field d/dx on the plane
# (black arrows, all equal) has polar components (cos phi, -sin phi / r) that vary from point to point (right
# panels); its covariant derivative vanishes, because the Christoffel symbols of the polar metric cancel the
# variation of the components. Output: figures/covariant_polar.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "covariant_polar.svg")
left = Graphics()
for rk in [0.6, 1.2, 1.8]:
    left += circle((0, 0), rk, color="lightgrey")
for k in range(8):
    left += line([(0, 0), (2.1 * cos(k * pi / 4), 2.1 * sin(k * pi / 4))], color="lightgrey")
for rk in [0.6, 1.2, 1.8]:
    for k in range(8):
        P = (rk * cos(k * pi / 4), rk * sin(k * pi / 4))
        left += arrow(P, (P[0] + 0.3, P[1]), color="black", width=1)
left += text(r"$X = \partial_x$: the same arrow everywhere", (0, 2.5), fontsize=10)
left.axes(False); left.set_aspect_ratio(1)
r, ph = var("r ph", domain="real")
M = Manifold(2, "R^2", structure="Riemannian")
cart = M.chart("x y"); x, y = cart[:]
U = M.open_subset("U"); pol = U.chart(r"r:(0,+oo) ph:(0,2*pi):\varphi")
rM, phM = pol[:]
tr = pol.transition_map(cart.restrict(U), (rM * cos(phM), rM * sin(phM))); tr.set_inverse(sqrt(x**2 + y**2), atan2(y, x), check=False)
g = M.metric("g"); g[0, 0] = 1; g[1, 1] = 1
X = M.vector_field(1, 0, name="X")                       # d/dx
XU = X.restrict(U); XU.display(pol.frame(), pol)
Xr, Xph = XU.comp(pol.frame(), pol)[0].expr(), XU.comp(pol.frame(), pol)[1].expr()
nab = g.connection()
DX = nab(X).restrict(U)
DX.display(pol.frame(), pol)
middle = plot(cos(ph), (ph, 0, 2 * pi), color="firebrick", legend_label=r"$X^r=\cos\varphi$") + plot(-sin(ph), (ph, 0, 2 * pi), color="royalblue", legend_label=r"$r X^\varphi=-\sin\varphi$")
middle += text(r"polar components along the circle $r=1$", (pi, 1.3), fontsize=10)
middle.set_legend_options(loc="lower left", font_size=8)
g2 = graphics_array([left, middle]); g2.save(OUT, figsize=[8.5, 3.8])
print("[PASS] polar components of d/dx are (cos phi, -sin phi / r):", bool((Xr - cos(phM)).simplify_full() == 0 and (Xph + sin(phM) / rM).simplify_full() == 0))
print("[PASS] the covariant derivative of d/dx vanishes in polar coordinates, although its components vary:", all(DX.comp(pol.frame(), pol)[i, j].expr().simplify_full() == 0 for i in range(2) for j in range(2)))
print("wrote", os.path.relpath(OUT))
