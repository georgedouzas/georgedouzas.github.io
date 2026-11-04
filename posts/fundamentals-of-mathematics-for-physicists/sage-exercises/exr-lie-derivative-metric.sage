# Exercise exr-lie-derivative-metric: Lie derivatives of the polar metric with SageManifolds' lie_derivative: along
# the rotation field d/dphi (zero, an isometry) and along the radial field r d/dr (twice the metric, a dilation).
M = Manifold(2, "R^2", structure="Riemannian"); U = M.open_subset("U")
pol = U.chart(r"r:(0,+oo) ph:(0,2*pi):\varphi"); r, ph = pol[:]
g = U.metric("g"); g[0, 0] = 1; g[1, 1] = r**2
rot = U.vector_field(0, 1, name="rot")           # d/dphi
rad = U.vector_field(r, 0, name="rad")           # r d/dr
print("L_rot g =", g.lie_derivative(rot).display())
print("L_rad g =", g.lie_derivative(rad).display(), " = 2 g:", g.lie_derivative(rad) == 2 * g)
f = U.scalar_field(r**2 * cos(ph), name="f")
print("L_rad f = rad(f) =", f.lie_derivative(rad).expr(), "; L_rot f =", f.lie_derivative(rot).expr())
Y = U.vector_field(1, 0, name="Y")               # d/dr
print("L_rot Y = [rot, Y] =", Y.lie_derivative(rot).display(), "; L_rad Y = [rad, Y] =", Y.lie_derivative(rad).display())
