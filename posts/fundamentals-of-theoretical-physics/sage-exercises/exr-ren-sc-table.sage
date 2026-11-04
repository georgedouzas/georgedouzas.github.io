# Exercise exr-ren-sc-table: the strong-coupling scale Lambda_s R = (l_D/(g4^2 (2 pi)^delta))^(1/delta) for delta = 1..6, alpha_4 = 1/30.
g4sq = 4 * pi / 30
for d in range(1, 7):
    D = 4 + d; ell = (4 * pi)^(D / 2) * gamma(D / 2)
    print("delta = %d: loop factor %.4g, Lambda_s R = %.3g" % (d, ell.n(), ((ell / (g4sq * (2 * pi)^d))^(1 / d)).n()))
