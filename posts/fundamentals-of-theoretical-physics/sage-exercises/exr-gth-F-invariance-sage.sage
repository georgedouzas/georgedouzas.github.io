# Exercise exr-gth-F-invariance-sage: for a general four-potential A_mu(t, x, y, z) and gauge parameter alpha, the
# six components of F_{mu nu} = d_mu A_nu - d_nu A_mu are unchanged by A_mu -> A_mu + d_mu alpha; and a concrete case.
t, x, y, z = var("t x y z", domain="real"); X = [t, x, y, z]
A = [function("A%d" % k)(t, x, y, z) for k in range(4)]; al = function("al")(t, x, y, z)
F = lambda P: matrix(SR, 4, 4, lambda m, n: diff(P[n], X[m]) - diff(P[m], X[n]))
A2 = [A[k] + diff(al, X[k]) for k in range(4)]
print("[PASS] all components of F are gauge invariant (mixed partial derivatives commute):", (F(A2) - F(A)).is_zero())
Ac = [0, t * x, 0, 0]; alc = sin(t * x)
A2c = [Ac[k] + diff(alc, X[k]) for k in range(4)]
print("concrete case A = (0, t x, 0, 0), alpha = sin(t x): A' =", A2c)
print("F_tx before:", F(Ac)[0, 1], "; after:", F(A2c)[0, 1].simplify_full())
print("[PASS] the concrete field strength is unchanged:", (F(A2c) - F(Ac)).apply_map(lambda e: e.simplify_full()).is_zero())
