# Exercise exr-vcr-frg-s4: the round sphere S^n of radius a in hyperspherical angles is an Einstein space with
# R_mn = (n-1)/a^2 gamma_mn (positive-definite metric gamma) and R = n(n-1)/a^2, for n = 2, ..., 5. In the signature
# (+, -, ..., -) of the book the internal metric is h = -gamma and R_mn = -(n-1)/a^2 h_mn.
a = var("a", domain="positive")
for n in range(2, 6):
    S = Manifold(n, "S%d" % n); X = S.chart(" ".join("c%d" % k for k in range(n))); c = X[:]
    gam = S.metric("gam"); w = 1
    for k in range(n):
        gam[k, k] = a ^ 2 * w; w = w * sin(c[k]) ^ 2
    Ric = gam.ricci()
    ok = all(bool((Ric[i, j].expr() - (n - 1) / a ^ 2 * gam[i, j].expr()).simplify_full() == 0) for i in range(n) for j in range(n))
    print("S^%d: R_mn = (n-1)/a^2 gamma_mn: %s; R = %s" % (n, ok, gam.ricci_scalar().expr().simplify_full()))
