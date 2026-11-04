# Exercise exr-tf-mixed-polar: the polar components of the (1, 1) tensor field T = d_x (x) dy, by SageManifolds'
# change of frame, compared with the transformation law; its trace is 0 in both charts.
M = Manifold(2, "R^2"); cart = M.chart("x y"); x, y = cart[:]
U = M.open_subset("U"); pol = U.chart(r"r:(0,+oo) ph:(0,2*pi):\varphi"); r, ph = pol[:]
tr = pol.transition_map(cart.restrict(U), (r*cos(ph), r*sin(ph))); tr.set_inverse(sqrt(x**2 + y**2), atan2(y, x), check=False)
fc, fp = cart.frame().restrict(U), pol.frame()
T = M.tensor_field(1, 1, name="T"); T[cart.frame(), 0, 1] = 1        # T^x_y = 1
TU = T.restrict(U); TU.display(fp, pol)
c = TU.comp(fp, pol)
pred = [[sin(ph)*cos(ph), r*cos(ph)**2], [-sin(ph)**2/r, -sin(ph)*cos(ph)]]
print("polar components T^a_b:"); print(matrix(SR, 2, 2, lambda a, b: c[a, b].expr()))
print("agree with T^r_r = sin cos, T^r_phi = r cos^2, T^phi_r = -sin^2/r, T^phi_phi = -sin cos:", all(bool((c[a, b].expr() - pred[a][b]).simplify_full() == 0) for a in range(2) for b in range(2)))
print("trace in Cartesian:", T.trace().expr(cart), "; trace in polar:", (c[0, 0].expr() + c[1, 1].expr()).simplify_full())
