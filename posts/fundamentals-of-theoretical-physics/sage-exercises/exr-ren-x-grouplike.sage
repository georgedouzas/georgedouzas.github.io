# Exercise exr-ren-x-grouplike: G(L) = sum alpha^n F_R[b_n](L) for the toy-model bamboos, compared with exp(L gamma(alpha)),
# gamma(alpha) = -arcsin(pi alpha)/pi.
var('e L a')
Fb = lambda n, s: prod(pi / sin(pi * j * e) for j in range(1, n + 1)) * s^(-n * e)
SR, FRs = {}, {0: 1}
for n in range(1, 5):
    SR[n] = -(Fb(n, 1) + sum(SR[k] * Fb(n - k, 1) for k in range(1, n)))
    FRs[n] = (Fb(n, exp(L)) + sum(SR[k] * Fb(n - k, exp(L)) for k in range(1, n)) + SR[n]).series(e, 1).truncate().coefficient(e, 0).simplify_log().expand()
G = sum(a^n * FRs[n] for n in range(5))
gam = taylor(-arcsin(pi * a) / pi, a, 0, 4)
print("gamma(alpha) =", gam)
print("G - exp(L gamma) to order alpha^4:", taylor(G - exp(L * gam), a, 0, 4).simplify_full())
print("coefficient of L in G:", G.coefficient(L, 1), "   of L^2:", G.coefficient(L, 2))
