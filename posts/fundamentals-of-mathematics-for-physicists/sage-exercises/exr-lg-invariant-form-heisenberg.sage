# Exercise exr-lg-invariant-form-heisenberg: the Maurer-Cartan form g^{-1} dg of the Heisenberg group has components
# sigma^X = dx, sigma^Y = dy, sigma^Z = dz - x dy along E_12, E_23, E_13; they are invariant under left translations,
# dual to X, Y, Z, and d sigma^Z = -sigma^X ^ sigma^Y.
M = Manifold(3, "H", structure="smooth"); C.<x, y, z> = M.chart()
g = matrix(SR, [[1, x, z], [0, 1, y], [0, 0, 1]])
for i, v in enumerate([x, y, z]):
    D = g.inverse() * diff(g, v)
    print(f"g^-1 dg/d{v} =", D.list())
sX = M.one_form(1, 0, 0, name="sX"); sY = M.one_form(0, 1, 0, name="sY"); sZ = M.one_form(0, -x, 1, name="sZ")
var("a b c")
Lh = M.diff_map(M, [x + a, y + b, z + c + a * y], name="L_h")
print("L_h^* sigma^Z =", Lh.pullback(sZ).display(), "; invariant:", Lh.pullback(sZ) == sZ)
X = M.vector_field(1, 0, 0); Y = M.vector_field(0, 1, x); Z = M.vector_field(0, 0, 1)
print("pairings sigma^A(B):", [[s(v).expr() for v in (X, Y, Z)] for s in (sX, sY, sZ)])
print("d sigma^Z =", sZ.exterior_derivative().display(), "; -sigma^X ^ sigma^Y =", (-sX.wedge(sY)).display(), "; equal:", sZ.exterior_derivative() == -sX.wedge(sY))
print("d sigma^X = 0, d sigma^Y = 0:", sX.exterior_derivative() == 0, sY.exterior_derivative() == 0)
