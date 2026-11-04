# Exercise exr-fld-spinor-boost: the spinor boost Lambda_s = exp(w S^{01}) in the Weyl representation, signature
# (+, -, -, -): it equals cosh(w/2) + 2 S^{01} sinh(w/2), it is not unitary, and it reproduces the vector boost
# through Lambda_s^{-1} gamma^mu Lambda_s = Lambda^mu_nu gamma^nu with Lambda = [[cosh w, sinh w], [sinh w, cosh w]].
# (cosh and sinh are written through exponentials so that Sage compares like with like.)
w = var("w", domain="real")
ch = lambda a: (exp(a) + exp(-a)) / 2; sh = lambda a: (exp(a) - exp(-a)) / 2
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z, one = zero_matrix(SR, 2), identity_matrix(SR, 2)
g = [block_matrix([[Z, one], [one, Z]], subdivide=False)] + [block_matrix([[Z, s[i]], [-s[i], Z]], subdivide=False) for i in range(3)]
S01 = (g[0] * g[1] - g[1] * g[0]) / 4
Ls = (w * S01).exp().simplify_full()
print("S^{01} = -(1/2) diag(sigma_1, -sigma_1):", S01 == -block_matrix([[s[0], Z], [Z, -s[0]]], subdivide=False) / 2, "; (2 S^{01})^2 = 1:", (2 * S01)**2 == identity_matrix(SR, 4))
print("exp(w S^{01}) = cosh(w/2) + 2 S^{01} sinh(w/2):", bool((Ls - (ch(w / 2) * identity_matrix(SR, 4) + 2 * S01 * sh(w / 2))).simplify_full() == 0))
print("Lambda_s^dagger Lambda_s = cosh(w) 1 + 2 sinh(w) S^{01}:", bool((Ls.conjugate_transpose() * Ls - (ch(w) * identity_matrix(SR, 4) + 2 * sh(w) * S01)).simplify_full() == 0), "(not 1: boosts are not unitary)")
Lam = identity_matrix(SR, 4); Lam[0, 0] = Lam[1, 1] = ch(w); Lam[0, 1] = Lam[1, 0] = sh(w)
Li = (-w * S01).exp().simplify_full()
print("Lambda_s^{-1} = exp(-w S^{01}):", bool((Li * Ls).simplify_full() == identity_matrix(SR, 4)))
print("Lambda_s^{-1} gamma^mu Lambda_s = Lambda^mu_nu gamma^nu:", all(bool((Li * g[a] * Ls - sum(Lam[a, b] * g[b] for b in range(4))).simplify_full() == 0) for a in range(4)))
print("the vector boost preserves eta:", bool((Lam.transpose() * diagonal_matrix([1, -1, -1, -1]) * Lam).simplify_full() == diagonal_matrix([1, -1, -1, -1])))
