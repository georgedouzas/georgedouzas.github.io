# Exercise exr-zeta-regularised-sum: the regularised sums sum e^{-eps n} and sum n e^{-eps n} in closed form, expanded
# in eps; the finite parts are zeta(0) = -1/2 and zeta(-1) = -1/12.
eps = var("eps", domain="positive"); n = var("n")
s0 = sum(exp(-eps*n), n, 1, oo).simplify_full()
s1 = sum(n*exp(-eps*n), n, 1, oo).simplify_full()
print("sum e^{-eps n} =", s0, "   expansion:", s0.series(eps == 0, 3))
print("sum n e^{-eps n} =", s1, "   expansion:", s1.series(eps == 0, 3))
print("finite parts:", s0.series(eps == 0, 3).truncate().coefficient(eps, 0), "= zeta(0) =", zeta(0), ";", s1.series(eps == 0, 3).truncate().coefficient(eps, 0), "= zeta(-1) =", zeta(-1))
