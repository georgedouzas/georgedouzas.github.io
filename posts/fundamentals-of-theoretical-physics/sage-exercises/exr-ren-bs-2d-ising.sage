# Exercise exr-ren-bs-2d-ising: the fixed point of K' = (3/8) log cosh 4K, its slope and the estimate of nu with b = sqrt 2.
K = var('K')
M = 3 / 8 * log(cosh(4 * K))
Ks = find_root(M - K, 0.3, 0.8)
lam = diff(M, K).subs(K=Ks)
print("K* = %.4f (exact K_c = %.4f), slope %.4f, nu = %.4f (exact 1)" % (Ks, (asinh(1) / 2).n(), lam.n(), (log(sqrt(2)) / log(lam)).n()))
print("relative error of K*: %.1f%%" % (100 * (Ks - asinh(1).n() / 2) / (asinh(1).n() / 2)))
