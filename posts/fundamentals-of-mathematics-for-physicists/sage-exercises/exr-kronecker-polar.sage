# Exercise exr-kronecker-polar: the Kronecker delta is the identity tensor field: its components are the identity
# matrix in every chart, here Cartesian and polar, with SageManifolds; and the metric contracted with its inverse
# gives the dimension.
M = Manifold(2, "R^2", structure="Riemannian")
cart = M.chart("x y"); x, y = cart[:]
U = M.open_subset("U"); pol = U.chart(r"r:(0,+oo) ph:(0,2*pi):\varphi"); r, ph = pol[:]
tr = pol.transition_map(cart.restrict(U), (r*cos(ph), r*sin(ph))); tr.set_inverse(sqrt(x**2 + y**2), atan2(y, x), check=False)
Id = M.tangent_identity_field()
print("identity tensor in Cartesian coordinates:", Id[cart.frame(), :])
print("in polar coordinates:", Id.restrict(U)[pol.frame(), :, pol])
print("its trace (contraction of the two indices):", Id.trace().expr())
g = M.metric("g"); g[0, 0] = 1; g[1, 1] = 1
ginv = g.inverse()
print("g^{ij} g_{jk} = delta^i_k:", ginv.contract(1, g, 0) == Id)
print("full contraction g^{ij} g_{ij} = n:", g.contract(0, 1, ginv, 0, 1).expr())
