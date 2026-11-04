# Exercise exr-polar-metric: the polar metric of the plane with SageManifolds: the Euclidean metric declared in
# Cartesian coordinates and displayed in polar coordinates, its determinant, inverse and volume element.
M = Manifold(2, "R^2", structure="Riemannian")
cart = M.chart("x y"); x, y = cart[:]
U = M.open_subset("U"); pol = U.chart(r"r:(0,+oo) ph:(0,2*pi):\varphi"); r, ph = pol[:]
tr = pol.transition_map(cart.restrict(U), (r*cos(ph), r*sin(ph))); tr.set_inverse(sqrt(x**2 + y**2), atan2(y, x), check=False)
g = M.metric("g"); g[0, 0] = 1; g[1, 1] = 1                      # Euclidean metric in Cartesian coordinates
gU = g.restrict(U)
print("metric in polar coordinates:", gU.display(pol.frame(), pol))
print("components:", gU[pol.frame(), :, pol])
print("determinant:", gU.determinant(pol.frame()).expr(pol))
print("inverse metric:", gU.inverse().display(pol.frame(), pol))
print("volume element sqrt(det g) =", sqrt(gU.determinant(pol.frame()).expr(pol)).simplify_full(), " (the Jacobian r of polar coordinates)")
