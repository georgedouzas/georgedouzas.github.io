# Exercise exr-elementary-evaluate: the elementary tensor u (x) v as a bilinear map on covectors, (u (x) v)(a, b) =
# a(u) b(v), evaluated on a = (1, 0, -1), b = (0, 1, 1), and the rank of its component matrix.
V = FiniteRankFreeModule(QQ, 3, name="V")
e = V.basis("e")
u = V([1, 2, 3], basis=e, name="u"); v = V([1, 0, -1], basis=e, name="v")
a = V.linear_form(name="a"); a[:] = [1, 0, -1]
b = V.linear_form(name="b"); b[:] = [0, 1, 1]
t = u * v
print("(u (x) v)(a, b) =", t(a, b), "; a(u) b(v) =", a(u) * b(v), "; equal:", t(a, b) == a(u) * b(v))
print("components u^i v^j:", t.comp(e)[:], "; rank of the component matrix:", matrix(QQ, t.comp(e)[:]).rank())
