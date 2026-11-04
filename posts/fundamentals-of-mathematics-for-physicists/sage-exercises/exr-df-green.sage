# Exercise exr-df-green: Stokes' theorem on the unit disc for alpha = -y^3 dx + x^3 dy: the integral of d alpha over
# the disc and the integral of alpha around the circle are both 3 pi/2.
M = Manifold(2, "R^2"); cart = M.chart("x y"); x, y = cart[:]
U = M.open_subset("U"); pol = U.chart(r"r:(0,+oo) ph:(0,2*pi):\varphi"); r, ph = pol[:]
tr = pol.transition_map(cart.restrict(U), (r*cos(ph), r*sin(ph))); tr.set_inverse(sqrt(x**2 + y**2), atan2(y, x), check=False)
fc, fp = cart.frame().restrict(U), pol.frame()
al = M.one_form(-y**3, x**3, name="alpha")
da = al.exterior_derivative()
print("d alpha =", da.display())
da.restrict(U).display(fp, pol)
dap = da.restrict(U).comp(fp, pol)[0, 1].expr().simplify_full()
print("d alpha in polar coordinates: (", dap, ") dr ^ dphi")
print("integral of d alpha over the disc:", integrate(integrate(dap, r, 0, 1), ph, 0, 2*pi))
S1 = Manifold(1, "I"); T = S1.chart("t:(0,2*pi)"); t = T[0]
gamma = S1.diff_map(M, {(T, cart): [cos(t), sin(t)]})
print("integral of alpha around the circle:", integrate(gamma.pullback(al)[0].expr(), t, 0, 2*pi))
