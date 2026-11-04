# Exercise exr-nonabelian-gauge: the gauge transformation of A^a_mu induced by the internal diffeomorphism
# y^m -> y^m + eps^a(x) K^m_a(y). Compute the Lie derivative of the mixed component g_{mu m} = K_{m a} A^a_mu of the
# ansatz along xi = eps^a(x) K_a and compare with K_{m c} delta A^c_mu for delta A^c = d_mu eps^c + eps_abc A^a_mu eps^b.
# The internal block is h = -gamma (signature (+,-,-,-,-,-)); both sides are linear in h, so its sign is immaterial.
th, ph = var("th ph")
K = [vector([-sin(ph), -cot(th) * cos(ph)]), vector([cos(ph), -cot(th) * sin(ph)]), vector([0, 1])]
gam = diagonal_matrix([1, sin(th)^2]); h = -gam; Klow = [h * K[a] for a in range(3)]        # K_{m a} = h_{mn} K^n_a
def lc(a, b, c): return Permutation([a + 1, b + 1, c + 1]).sign() if len({a, b, c}) == 3 else 0
A = [var("A%d" % a) for a in range(3)]; eps = [var("e%d" % a) for a in range(3)]; deps = [var("de%d" % a) for a in range(3)]   # one spacetime index mu fixed
ghat = [sum(Klow[a][m] * A[a] for a in range(3)) for m in range(2)]                      # g_{mu m}
xi = [sum(eps[a] * K[a][m] for a in range(3)) for m in range(2)]                         # xi^m
dxi = [sum(deps[a] * K[a][m] for a in range(3)) for m in range(2)]                       # d_mu xi^m
Lg = [sum(xi[n] * diff(ghat[m], [th, ph][n]) for n in range(2)) + sum(dxi[n] * h[n, m] for n in range(2)) + sum(diff(xi[n], [th, ph][m]) * ghat[n] for n in range(2)) for m in range(2)]
dA = [deps[c] + sum(lc(a, b, c) * A[a] * eps[b] for a in range(3) for b in range(3)) for c in range(3)]
print("delta A^c =", dA)
print("[PASS] L_xi g_{mu m} = K_{m c} delta A^c_mu with delta A^c = d_mu eps^c + eps_abc A^a eps^b:", all(bool((Lg[m] - sum(Klow[c][m] * dA[c] for c in range(3))).simplify_full() == 0) for m in range(2)))
Lh = [[sum(xi[n] * diff(h[m, p], [th, ph][n]) for n in range(2)) + sum(diff(xi[n], [th, ph][m]) * h[n, p] + diff(xi[n], [th, ph][p]) * h[m, n] for n in range(2)) for p in range(2)] for m in range(2)]
print("[PASS] the internal block is unchanged, L_xi h = 0 (the K_a are Killing):", all(bool(Lh[m][p].simplify_full() == 0) for m in range(2) for p in range(2)))
print("[PASS] with f_ab^c = -eps_abc from the brackets, delta A^c = d eps^c - f_ab^c A^a eps^b (Duff, eq. 30):", all(bool((dA[c] - (deps[c] - sum(-lc(a, b, c) * A[a] * eps[b] for a in range(3) for b in range(3)))).simplify_full() == 0) for c in range(3)))
