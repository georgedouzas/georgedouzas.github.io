# Exercise exr-ren-kk-xdelta: lattice points of Z^delta in a ball of radius N against X_delta N^delta.
import itertools
for d in [1, 2, 3]:
    X = pi^(d / 2) / gamma(1 + d / 2)
    for N in [10, 30]:
        cnt = sum(1 for v in itertools.product(range(-N, N + 1), repeat=d) if sum(x * x for x in v) <= N * N)
        print("delta = %d, N = %2d: count %6d, X_delta N^delta = %.1f" % (d, N, cnt, (X * N^d).n()))
