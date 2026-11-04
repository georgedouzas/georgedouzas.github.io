# Exercise exr-zr-sage-values: exact zeta values at 0, -1, -3, zeta'(0) against -log(2 pi)/2, the finite parts of cutoff
# sums, and the zeta-regularised determinant of -d^2/dx^2 on (0, pi) with Dirichlet conditions.
print("zeta(-1), zeta(0), zeta(-3) =", zeta(-1), zeta(0), zeta(-3))
s = var("s")
zp = diff(zeta(s), s).subs(s=0).n(digits=30)
print("zeta'(0) =", zp, "; -log(2 pi)/2 =", (-log(2 * pi) / 2).n(digits=30))
e = var("epsilon", domain="positive"); n = var("n", domain="integer")
print("sum e^{-eps n} =", taylor(sum(exp(-e * n), n, 1, oo), e, 0, 2))
print("sum n^3 e^{-eps n} =", taylor(sum(n ** 3 * exp(-e * n), n, 1, oo), e, 0, 2))
print("det_zeta = e^{-2 zeta'(0)} =", exp(-2 * diff(zeta(s), s).subs(s=0)).n(digits=20), "; 2 pi =", (2 * pi).n(digits=20))
