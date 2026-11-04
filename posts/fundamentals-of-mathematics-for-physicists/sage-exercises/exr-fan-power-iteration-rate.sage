# Exercise exr-fan-power-iteration-rate: Rayleigh quotients of the power iteration for A = diag(l1, l2), x0 = (a, b).
l1, l2, a, b, n = 2, -1.5, 1.0, 1.0, None
A = diagonal_matrix(RDF, [l1, l2]); x0 = vector(RDF, [a, b])
for n in (1, 5, 10, 20, 30):
    xn = A^n * x0
    lam = (xn * (A * xn)) / (xn * xn)
    print("n = %2d: lambda^(n) - l1 = %+.3e, (l2/l1)^(2n) = %.3e, ratio %.4f" % (n, lam - l1, (l2 / l1)^(2 * n), (lam - l1) / (l2 / l1)^(2 * n)))
print("predicted ratio l1 (b/a)^2 (l2/l1 - 1) =", l1 * (b / a)^2 * (l2 / l1 - 1))
