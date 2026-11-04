# Exercise exr-of-rotation-form-polar: the one-form omega = -y dx + x dy is r^2 dphi in polar coordinates, and its
# integral around the unit circle, computed as the pullback by gamma(t) = (cos t, sin t), is 2 pi.
M = Manifold(2, "R^2"); cart = M.chart("x y"); x, y = cart[:]
U = M.open_subset("U"); pol = U.chart(r"r:(0,+oo) ph:(0,2*pi):\varphi"); r, ph = pol[:]
tr = pol.transition_map(cart.restrict(U), (r*cos(ph), r*sin(ph))); tr.set_inverse(sqrt(x**2 + y**2), atan2(y, x), check=False)
fc, fp = cart.frame().restrict(U), pol.frame()
om = M.one_form(-y, x, name="omega")
print("omega in polar coordinates:", om.restrict(U).display(fp, pol))
S1 = Manifold(1, "I"); T = S1.chart("t:(0,2*pi)"); t = T[0]
gamma = S1.diff_map(M, {(T, cart): [cos(t), sin(t)]}, name="gamma")
pb = gamma.pullback(om)
print("pullback gamma^* omega =", pb.display(), "; integral over (0, 2 pi):", integrate(pb[0].expr(), t, 0, 2*pi))
print("omega is not closed, d omega =", om.exterior_derivative().display())
