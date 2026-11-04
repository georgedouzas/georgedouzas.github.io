# Exercise exr-raise-index: raising and lowering indices with the round metric in SageManifolds (up() and down()).
S2 = Manifold(2, "S^2", structure="Riemannian"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]; assume(sin(th) > 0)
g = A.metric("g"); g[0, 0] = 1; g[1, 1] = sin(th)**2
Z = A.scalar_field(cos(th), name="Z")
dZ = Z.differential()
gradZ = dZ.up(g)
print("dZ =", dZ.display(), " -> gradient (index raised):", gradZ.display())
print("length of the gradient:", sqrt(g(gradZ, gradZ).expr()).simplify_full())
dph = A.vector_field(0, 1, name="dph")
print("lowering d/dphi:", dph.down(g).display())
print("raising then lowering d/dphi returns it:", dph.down(g).up(g) == dph)
print("the inverse metric:", g.inverse().display())
