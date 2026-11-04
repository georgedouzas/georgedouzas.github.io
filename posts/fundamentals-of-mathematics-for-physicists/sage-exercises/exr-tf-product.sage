# Exercise exr-tf-product: the tensor product of the round metric with itself is a type (0, 4) tensor field with 16
# components, of which 4 are non-zero; and the tensor product of one-forms is not commutative.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
g = A.tensor_field(0, 2, name="g"); g[0, 0] = 1; g[1, 1] = sin(th)**2
gg = g*g
print("type of g (x) g:", gg.tensor_type())
nz = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2) if gg[i, j, k, l].expr() != 0]
print("non-zero components (index lists, 0 = theta, 1 = phi):", [(I, gg[I].expr()) for I in nz])
dth, dph = sph.coframe()[0], sph.coframe()[1]
print("dtheta (x) dphi = dphi (x) dtheta:", dth*dph == dph*dth, "; their difference is dtheta ^ dphi:", dth*dph - dph*dth == dth.wedge(dph))
