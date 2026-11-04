# Exercise exr-sr-field-complex: under the active boost exp(eta K1), E - iB transforms by exp(-i eta L1) and E + iB by
# exp(+i eta L1), with (L1)_{jk} = -eps_{1jk}: the representations (1,0) and (0,1).
var('eta', domain='real')
E = vector(SR, var('E1 E2 E3')); B = vector(SR, var('B1 B2 B3'))
eps = lambda i, j, k: sign(Permutation([i + 1, j + 1, k + 1]).signature()) if len({i, j, k}) == 3 else 0
F = matrix(SR, 4, 4)
for i in range(3):
    F[i + 1, 0] = E[i]; F[0, i + 1] = -E[i]
    for j in range(3):
        F[i + 1, j + 1] = -sum(eps(i, j, k)*B[k] for k in range(3))
K1 = matrix(SR, 4, 4); K1[0, 1] = K1[1, 0] = 1
L = (eta*K1).exp().simplify_full()
Fp = L*F*L.transpose()
Ep = vector([Fp[i + 1, 0] for i in range(3)]); Bp = vector([-Fp[2, 3], -Fp[3, 1], -Fp[1, 2]])
L1 = matrix(SR, 3, 3, lambda j, k: -eps(0, j, k))
Rm = (-I*eta*L1).exp().simplify_full(); Rp = (I*eta*L1).exp().simplify_full()
print("E - iB -> exp(-i eta L1)(E - iB):", ((Ep - I*Bp) - Rm*(E - I*B)).simplify_full() == 0)
print("E + iB -> exp(+i eta L1)(E + iB):", ((Ep + I*Bp) - Rp*(E + I*B)).simplify_full() == 0)
print("E'^2 - B'^2 - (E^2 - B^2) =", (Ep*Ep - Bp*Bp - E*E + B*B).simplify_full(), "   E'.B' - E.B =", (Ep*Bp - E*B).simplify_full())
