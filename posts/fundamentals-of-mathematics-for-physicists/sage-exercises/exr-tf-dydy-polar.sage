# Exercise exr-tf-dydy-polar: the polar components of dy (x) dy and of dx (x) dy, and the sum
# dx (x) dx + dy (x) dy = dr (x) dr + r^2 dphi (x) dphi, the Euclidean metric.
M = Manifold(2, "R^2"); cart = M.chart("x y"); x, y = cart[:]
U = M.open_subset("U"); pol = U.chart(r"r:(0,+oo) ph:(0,2*pi):\varphi"); r, ph = pol[:]
tr = pol.transition_map(cart.restrict(U), (r*cos(ph), r*sin(ph))); tr.set_inverse(sqrt(x**2 + y**2), atan2(y, x), check=False)
fc, fp = cart.frame().restrict(U), pol.frame()
dx, dy = cart.coframe()[0], cart.coframe()[1]
for name, T in [("dy (x) dy", dy*dy), ("dx (x) dy", dx*dy), ("dx (x) dx + dy (x) dy", dx*dx + dy*dy)]:
    TU = T.restrict(U); TU.display(fp, pol)
    c = TU.comp(fp, pol)
    print(name, "in polar coordinates:", matrix(SR, 2, 2, lambda a, b: c[a, b].expr().simplify_full()).list())
