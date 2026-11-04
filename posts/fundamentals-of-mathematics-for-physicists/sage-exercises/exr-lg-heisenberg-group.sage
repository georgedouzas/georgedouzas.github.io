# Exercise exr-lg-heisenberg-group: the Heisenberg group of upper unitriangular 3x3 matrices h(x, y, z): the product
# and the inverse are polynomial in the coordinates (so the group is a Lie group of dimension 3), and it is not abelian.
x1, y1, z1, x2, y2, z2 = var("x1 y1 z1 x2 y2 z2", domain="real")
def h(x, y, z): return matrix(SR, [[1, x, z], [0, 1, y], [0, 0, 1]])
P = h(x1, y1, z1) * h(x2, y2, z2)
print("h(x1,y1,z1) h(x2,y2,z2) =", P.list())
print("product is h(x1+x2, y1+y2, z1+z2+x1*y2):", P == h(x1 + x2, y1 + y2, z1 + z2 + x1 * y2))
Hi = h(x1, y1, z1).inverse().simplify_full()
print("inverse =", Hi.list(), "; equals h(-x, -y, x*y - z):", bool(Hi == h(-x1, -y1, x1 * y1 - z1)))
C = (h(x1, y1, z1) * h(x2, y2, z2) - h(x2, y2, z2) * h(x1, y1, z1))
print("g1 g2 - g2 g1 =", C.list(), "(nonzero: the group is not abelian; the commutator only moves z)")
