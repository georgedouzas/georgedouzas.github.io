# Exercise exr-kkf-anomaly: a Dirac mode of charge q is a left-handed Weyl fermion of charge q and one of charge -q; the
# sums of q and q^3 over the left-handed fermions of a truncated tower |n| <= N with twist alpha vanish identically.
alpha, R = var("alpha R")
for N in (1, 3, 10):
    qs = [(n + alpha) / R for n in range(-N, N + 1)]
    left = qs + [-q for q in qs]
    print("N = %d: sum q = %s, sum q^3 = %s" % (N, sum(left).simplify_full(), sum(q ** 3 for q in left).simplify_full()))
