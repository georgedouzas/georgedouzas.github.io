# Exercise exr-cq-lattice-modes: normal modes of a periodic chain of N = 6 sites with spacing a and mass m: the coupling
# matrix K = m^2 + (2 - shift - shift^{-1})/a^2 has eigenvalues m^2 + (4/a^2) sin^2(pi n/N).
N = 6; a, m = var("a m", domain="positive")
K = matrix(SR, N, N, lambda i, j: m ** 2 + 2 / a ** 2 if i == j else (-1 / a ** 2 if (i - j) % N in (1, N - 1) else 0))
ev = sorted([e.simplify_full() for e in K.eigenvalues()], key=lambda e: e.subs(a=1, m=1))
pred = sorted([(m ** 2 + 4 / a ** 2 * sin(pi * n / N) ** 2).simplify_full() for n in range(N)], key=lambda e: e.subs(a=1, m=1))
print("eigenvalues:", ev)
print("agree with m^2 + (4/a^2) sin^2(pi n/N):", all(bool(u == v) for u, v in zip(ev, pred)))
n = var("n")
print("small-n expansion of (2/a) sin(pi n/N) in n:", (2 / a * sin(pi * n / N)).series(n, 4), " vs k_n = 2 pi n/(N a)")
