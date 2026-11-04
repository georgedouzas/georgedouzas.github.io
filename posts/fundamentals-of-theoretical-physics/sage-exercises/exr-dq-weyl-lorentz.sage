# Exercise exr-dq-weyl-lorentz: for M = exp(-w sigma_3/2), M^dag sigma-bar^mu M = Lambda^mu_nu sigma-bar^nu and M^-1 sigma^mu (M^dag)^-1 =
# Lambda^mu_nu sigma^nu with the boost Lambda along z of rapidity w.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
w = var("w", domain="real")
M = diagonal_matrix(SR, [exp(-w / 2), exp(w / 2)])
sig = [one] + s; sigb = [one] + [-x for x in s]
ch, sh = (exp(w) + exp(-w)) / 2, (exp(w) - exp(-w)) / 2
Lam = matrix(SR, [[ch, 0, 0, sh], [0, 1, 0, 0], [0, 0, 1, 0], [sh, 0, 0, ch]])
z = lambda A_: (A_.apply_map(lambda e: e.expand().simplify_full())).is_zero()
print("M^dag sigma-bar^mu M = Lambda^mu_nu sigma-bar^nu:", all(z(M.conjugate_transpose() * sigb[mu] * M - sum(Lam[mu, nu] * sigb[nu] for nu in range(4))) for mu in range(4)))
print("M^-1 sigma^mu (M^dag)^-1 = Lambda^mu_nu sigma^nu:", all(z(M.inverse() * sig[mu] * M.conjugate_transpose().inverse() - sum(Lam[mu, nu] * sig[nu] for nu in range(4))) for mu in range(4)))
S = blk(M, Z, Z, M.conjugate_transpose().inverse())
S03 = (g[0] * g[3] - g[3] * g[0]) / 4
Sexp = (exp(w / 2) + exp(-w / 2)) / 2 * I4 + 2 * S03 * (exp(w / 2) - exp(-w / 2)) / 2    # (2 S^{03})^2 = 1
print("S = diag(M, (M^dag)^-1) equals exp(w S^{03}) = cosh(w/2) + 2 S^{03} sinh(w/2):", z(S - Sexp))
