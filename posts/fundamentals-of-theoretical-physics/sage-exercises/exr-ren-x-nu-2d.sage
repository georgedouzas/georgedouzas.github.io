# Exercise exr-ren-x-nu-2d: three estimates of nu for the two-dimensional Ising model.
K = var('K'); M = 3 / 8 * log(cosh(4 * K)); Ks = find_root(M - K, 0.3, 0.8)
nu_block = (log(sqrt(2)) / log(diff(M, K).subs(K=Ks))).n()
n, e = 1, 2
nu1 = 1 / 2 + (n + 2) * e / (4 * (n + 8)); nu2 = nu1 + (n + 2) * (n^2 + 23 * n + 60) * e^2 / (8 * (n + 8)^3)
print("block spin %.3f, epsilon expansion first order %.3f, second order %.3f, exact 1" % (nu_block, nu1, nu2))
