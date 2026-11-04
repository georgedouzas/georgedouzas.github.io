# Exercise exr-fld-bianchi: for an arbitrary potential A_mu(t, x, y, z) the field strength F = dA satisfies the
# Bianchi identity d_l F_mn + d_m F_nl + d_n F_lm = 0; with A_mu = (Phi, -A_vec), E_i = F_{0i}, F_{ij} = -eps_{ijk} B_k
# its components are div B = 0 and curl E + d_t B = 0. Signature (+, -, -, -).
X = var("t x y z", domain="real")
A = [function("A%d" % i)(*X) for i in range(4)]
F = [[diff(A[n], X[m]) - diff(A[m], X[n]) for n in range(4)] for m in range(4)]
ok = all(bool((diff(F[m][n], X[l]) + diff(F[n][l], X[m]) + diff(F[l][m], X[n])).simplify_full() == 0) for l in range(4) for m in range(4) for n in range(4))
print("Bianchi identity d_[l F_mn] = 0 for all 64 index triples:", ok)
E = vector([F[0][i] for i in (1, 2, 3)])
B = vector([-F[2][3], -F[3][1], -F[1][2]])            # F_{ij} = -eps_{ijk} B_k
Phi, Av = A[0], vector([-A[1], -A[2], -A[3]])
grad = lambda f: vector([diff(f, X[i]) for i in (1, 2, 3)])
curl = lambda V: vector([diff(V[2], X[2]) - diff(V[1], X[3]), diff(V[0], X[3]) - diff(V[2], X[1]), diff(V[1], X[1]) - diff(V[0], X[2])])
div = lambda V: sum(diff(V[i], X[i + 1]) for i in range(3))
print("E = -grad Phi - d_t A:", bool((E - (-grad(Phi) - diff(Av, X[0]))).simplify_full() == 0), "; B = curl A:", bool((B - curl(Av)).simplify_full() == 0))
print("div B = 0:", bool(div(B).simplify_full() == 0), "; curl E + d_t B = 0:", bool((curl(E) + diff(B, X[0])).simplify_full() == 0))
