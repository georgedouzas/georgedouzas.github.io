# Exercise exr-spd-orbifold-weights: invariant spinors of T^6/Z_N for rotations by 2 pi v in three complex planes. The lift
# eps * exp(2 pi i v.h) on the weight h is chosen with eps = +-1 such that its N-th power is 1 (if possible).
import itertools
for v, N in (((1/3, 1/3, -2/3), 3), ((1/4, 1/4, -1/2), 4), ((1/6, 1/3, -1/2), 6), ((1/3, 0, 0), 3)):
    W = list(itertools.product((1/2, -1/2), repeat=3))
    lifts = [eps for eps in (1, -1) if all(bool(exp(2 * pi * I * N * sum(vi * hi for vi, hi in zip(v, h))) * eps ^ N == 1) for h in W)]
    if not lifts:
        print("v =", v, ": no lift to spinors as a Z_%d action" % N); continue
    eps = lifts[0]
    inv = [h for h in W if bool(eps * exp(2 * pi * I * sum(vi * hi for vi, hi in zip(v, h))) == 1)]
    print("v = %s (sum %s), lift sign %+d: invariant weights %s, chiralities %s" % (v, sum(v), eps, inv, [(-1) ^ 3 * prod(2 * t for t in h) for h in inv]))
