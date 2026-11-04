# Exercise exr-of-dr-polar: the differential of r = sqrt(x^2 + y^2) on the punctured plane, in Cartesian and in polar
# coordinates, its values on the polar basis, and the chain rule d(r^2) = 2 r dr.
M = Manifold(2, "R^2"); cart = M.chart("x y"); x, y = cart[:]
U = M.open_subset("U"); pol = U.chart(r"r:(0,+oo) ph:(0,2*pi):\varphi"); r, ph = pol[:]
tr = pol.transition_map(cart.restrict(U), (r*cos(ph), r*sin(ph))); tr.set_inverse(sqrt(x**2 + y**2), atan2(y, x), check=False)
fc, fp = cart.frame().restrict(U), pol.frame()
R = U.scalar_field({pol: r}, name="r")
dR = R.differential()
print("dr in Cartesian coordinates:", dR.display(fc, cart.restrict(U)))
print("dr in polar coordinates:", dR.display(fp, pol))
print("dr(d_r) =", dR(fp[0]).expr(pol), "; dr(d_phi) =", dR(fp[1]).expr(pol))
print("d(r^2) = 2 r dr:", (R*R).differential() == 2*R*dR)
