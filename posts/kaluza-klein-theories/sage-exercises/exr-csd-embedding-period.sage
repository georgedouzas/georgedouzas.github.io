# Exercise exr-csdr-tau-exponential (script exr-csd-embedding-period): the one-parameter groups exp(alpha tau) in SU(3)
# and exp(alpha Q_3) in SU(2), their periods, and the value of exp(4 pi tau), a central element of SU(3).
al = var("al", domain="real")
tau = I / 3 * diagonal_matrix(SR, [1, 1, -2]); Q3 = diagonal_matrix(SR, [1, -1]) / (2 * I)
dexp = lambda D: diagonal_matrix(SR, [exp(al * d) for d in D.diagonal()])     # exponential of a diagonal matrix
et = dexp(tau); eq = dexp(Q3)
print("exp(alpha tau) =", et.diagonal(), "; exp(alpha Q_3) =", eq.diagonal())
print("[PASS] exp(alpha tau) is unitary with determinant 1:", bool((et * et.conjugate_transpose()).simplify_full() == identity_matrix(3)) and bool(et.det().simplify_full() == 1))
print("[PASS] periods: exp(alpha Q_3) = 1 first at alpha = 4 pi, exp(alpha tau) = 1 first at alpha = 6 pi:",
      eq.subs(al=4 * pi) == identity_matrix(2) and all(eq.subs(al=k * pi) != identity_matrix(2) for k in range(1, 4))
      and et.subs(al=6 * pi) == identity_matrix(3) and all(et.subs(al=k * pi) != identity_matrix(3) for k in range(1, 6)))
z = et.subs(al=4 * pi).simplify_full()
print("exp(4 pi tau) =", z.diagonal())
print("[PASS] exp(4 pi tau) = exp(4 pi i/3) 1, a non-trivial element of the centre Z_3 of SU(3):", bool((z - exp(4 * pi * I / 3) * identity_matrix(3)).simplify_full() == 0) and bool(exp(4 * pi * I / 3)^3 == 1))
