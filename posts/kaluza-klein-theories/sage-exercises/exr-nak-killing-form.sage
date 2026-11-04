# Exercise exr-nak-killing-form: the structure constants f_ab^c = -eps_abc of the sphere's Killing vectors satisfy the
# Jacobi identity, and the Killing form B_ab = f_ac^d f_bd^c of the algebra is -2 delta_ab, negative definite: the
# algebra is that of a compact group. The opposite constants +eps_abc of the gauge transformation give the same B.
def lc(a, b, c): return Permutation([a + 1, b + 1, c + 1]).sign() if len({a, b, c}) == 3 else 0
for sgn in [-1, 1]:
    f = lambda a, b, c: sgn * lc(a, b, c)
    jac = all(sum(f(a, b, d) * f(d, c, e) + f(b, c, d) * f(d, a, e) + f(c, a, d) * f(d, b, e) for d in range(3)) == 0 for a in range(3) for b in range(3) for c in range(3) for e in range(3))
    Bk = matrix(ZZ, 3, 3, lambda a, b: sum(f(a, c, d) * f(b, d, c) for c in range(3) for d in range(3)))
    print("f_ab^c = %+d eps_abc: Killing form" % sgn, Bk.list())
    print("[PASS] Jacobi identity and B_ab = -2 delta_ab for f = %+d eps:" % sgn, jac and Bk == -2 * identity_matrix(3))
