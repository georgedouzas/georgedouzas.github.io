# Exercise exr-kkr-weyl-maxwell: the Maxwell density sqrt|g| g^{mu rho} g^{nu sigma} F_{mu nu} F_{rho sigma} under
# g -> Omega^2 g picks up Omega^{n - 4} in n dimensions: it is Weyl invariant exactly in four dimensions. Checked with
# a generic symmetric metric (its inverse as free symbols) and a generic antisymmetric F for n = 2, ..., 5.
Om = var("Om"); assume(Om > 0)
for n in range(2, 6):
    g = matrix(SR, n, n, lambda i, j: var("g%d%d" % (min(i, j), max(i, j))))
    F = matrix(SR, n, n, lambda i, j: 0 if i == j else (var("F%d%d" % (i, j)) if i < j else -var("F%d%d" % (j, i))))
    gi = matrix(SR, n, n, lambda i, j: var("h%d%d" % (min(i, j), max(i, j))))   # g^{-1} as free symmetric symbols
    dens = lambda gi_: sum(gi_[a, c] * gi_[b, d] * F[a, b] * F[c, d] for a in range(n) for b in range(n) for c in range(n) for d in range(n))
    ratio_det = (Om^2 * g).det() / g.det()                       # = Om^{2n}, so sqrt ratio = Om^n
    ratio = (dens(gi / Om^2) / dens(gi)).simplify_full()
    print("n = %d: det ratio = %s, F^2 ratio = %s, density ratio = Om^%d" % (n, ratio_det.simplify_full(), ratio, n - 4))
    if n == 4:
        print("[PASS] in four dimensions sqrt|g| F^2 is Weyl invariant:", bool(ratio_det.simplify_full() == Om^8) and bool(ratio == Om^(-4)))
