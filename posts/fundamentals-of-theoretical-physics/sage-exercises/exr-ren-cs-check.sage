# Exercise exr-ren-cs-check: the one-loop four-point vertex function of the massive quartic theory satisfies the
# renormalisation group equation (mu d/dmu + beta d/dlambda + gamma_m m^2 d/dm^2 - 4 gamma_phi) Gamma = O(lambda^3).
var('t lam m2')
I = lambda r: sqrt(1 + 4 / r) * log((sqrt(1 + 4 / r) + 1) / (sqrt(1 + 4 / r) - 1)) - 2
BR = lambda P: -(1 / (16 * pi^2)) * (log(m2) - 2 * t + I(P / m2))          # Euclidean invariants, mu = e^t
D = lambda F: diff(F, t) + 3 * lam^2 / (16 * pi^2) * diff(F, lam) + lam * m2 / (16 * pi^2) * diff(F, m2)
for (S, T, U) in [(3, 4, 5), (0.5, 20, 7)]:
    Gam = lam - lam^2 / 2 * (BR(S) + BR(T) + BR(U))
    eq = D(Gam)
    print("S, T, U = %s: order lambda^2 part of the equation: %s;  order lambda^3 at m^2 = 1, t = 0: %.3e" % ((S, T, U), taylor(eq, lam, 0, 2).simplify_full(), eq.subs(m2=1, t=0).diff(lam, 3).subs(lam=0).n() / 6))
