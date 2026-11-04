# Exercise exr-covector-transform: under e'_i = A^j_i e_j the components of a vector transform with A^-1 and those of a
# covector with A, so the pairing a_i u^i is the same number in both bases.
V = FiniteRankFreeModule(QQ, 3, name="V")
e = V.basis("e")
u = V([1, 2, 3], basis=e, name="u")
a = V.linear_form(name="a"); a[:] = [1, 0, -1]
f = V.basis("f", from_family=(e[0] + e[1], e[1], e[2]))
print("components of u: in e", u.comp(e)[:], ", in f", u.comp(f)[:])
print("components of a: in e", a.comp(e)[:], ", in f", a.comp(f)[:])
print("a(u) = a_i u^i in e:", sum(a.comp(e)[i] * u.comp(e)[i] for i in range(3)), "; in f:", sum(a.comp(f)[i] * u.comp(f)[i] for i in range(3)), "; a(u) =", a(u))
A = matrix(QQ, [[1, 0, 0], [1, 1, 0], [0, 0, 1]])   # columns: f_j in the basis e
print("A =", A.rows(), "; u_f = A^-1 u_e:", list(A.inverse() * vector(QQ, u.comp(e)[:])) == list(u.comp(f)[:]), "; a_f = a_e A (row times A):", list(vector(QQ, a.comp(e)[:]) * A) == list(a.comp(f)[:]))
