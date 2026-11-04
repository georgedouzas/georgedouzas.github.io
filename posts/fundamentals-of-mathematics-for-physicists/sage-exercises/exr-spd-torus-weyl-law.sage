# Exercise exr-spd-torus-weyl-law: counting Dirac eigenvalues with |lambda| <= Lambda on the periodic T^2 and T^3 (R = 1)
# against Weyl's law N Vol omega_n Lambda^n / (2 pi)^n = N omega_n Lambda^n.
import itertools
for n, N, omega in ((2, 2, pi), (3, 2, 4 * pi / 3)):
    for L in (10, 20, 40):
        pts = sum(1 for m in itertools.product(range(-L, L + 1), repeat=n) if sum(x * x for x in m) <= L * L)
        weyl = (N * omega * L ^ n).n()
        print("T^%d, Lambda = %2d: count = %7d, Weyl = %10.1f, ratio = %.4f" % (n, L, N * pts, weyl, N * pts / weyl))
