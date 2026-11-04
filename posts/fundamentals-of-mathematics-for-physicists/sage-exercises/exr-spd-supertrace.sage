# Exercise exr-spd-supertrace: Tr(gamma_* exp(-t D^2)) on the unit sphere with M = 3: the levels k >= 1 cancel between the
# chiralities and the supertrace is the index 3 for every t; the ordinary trace grows like 2/t.
M = 3
for t in (0.05, 0.2, 1.0, 3.0):
    zero = M                                                       # k = 0: M modes of chirality +1
    plus = sum((M + 2 * k) * exp(-t * k * (k + M)) for k in range(1, 400)).n()
    minus = plus
    strace = zero + plus - minus
    trace = zero + plus + minus
    print("t = %4.2f: supertrace = %.10f, trace = %10.4f, 2/t = %.4f" % (t, strace, trace, 2 / t))
