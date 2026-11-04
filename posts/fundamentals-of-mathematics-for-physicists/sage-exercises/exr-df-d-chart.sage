# Exercise exr-df-d-chart: the exterior derivative does not depend on the chart: omega = -y dx + x dy has
# d omega = 2 dx ^ dy in Cartesian coordinates, and omega = r^2 dphi has d omega = 2 r dr ^ dphi in polar ones.
M = Manifold(2, "R^2"); cart = M.chart("x y"); x, y = cart[:]
U = M.open_subset("U"); pol = U.chart(r"r:(0,+oo) ph:(0,2*pi):\varphi"); r, ph = pol[:]
tr = pol.transition_map(cart.restrict(U), (r*cos(ph), r*sin(ph))); tr.set_inverse(sqrt(x**2 + y**2), atan2(y, x), check=False)
fc, fp = cart.frame().restrict(U), pol.frame()
om = M.one_form(-y, x, name="omega")
print("d omega in Cartesian coordinates:", om.exterior_derivative().display())
omp = U.one_form(name="omega_p"); omp[fp, :] = [0, r**2]
print("omega = r^2 dphi:", om.restrict(U) == omp)
print("d(r^2 dphi) =", omp.exterior_derivative().display(fp, pol))
print("the two computations agree:", omp.exterior_derivative() == om.exterior_derivative().restrict(U))
