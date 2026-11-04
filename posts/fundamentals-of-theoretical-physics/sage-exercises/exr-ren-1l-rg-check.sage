# Exercise exr-ren-1l-rg-check: mu independence of the one-loop pole mass and four-point amplitude of the quartic theory.
# With t = log mu, the total derivative is d/dt = partial_t + beta partial_lambda + gamma_m m^2 partial_{m^2}, where
# beta = 3 lambda^2/(16 pi^2) and gamma_m m^2 = lambda m^2/(16 pi^2).
var('t lam m2 x'); assume(m2 > 0)
D = lambda F: diff(F, t) + 3 * lam^2 / (16 * pi^2) * diff(F, lam) + lam * m2 / (16 * pi^2) * diff(F, m2)
mP2 = m2 * (1 + lam / (32 * pi^2) * (log(m2) - 2 * t - 1))          # log(m^2/mu^2) with mu = e^t
print("d m_P^2/dt =", D(mP2).expand().simplify_full(), "   (order lambda^2)")
I = lambda r: sqrt(1 + 4 / r) * log((sqrt(1 + 4 / r) + 1) / (sqrt(1 + 4 / r) - 1)) - 2   # int_0^1 log(1 + x(1-x) r) dx, r = P/m^2 > 0
BR = lambda P: -(1 / (16 * pi^2)) * (log(m2) - 2 * t + I(P / m2))
for (S, T, U) in [(2, 3, 5), (1, 10, 100)]:
    M = -lam + lam^2 / 2 * (BR(S) + BR(T) + BR(U))
    dM = D(M)
    print("S, T, U = %s: dM/dt to order lambda^2 = %s;  order lambda^3 coefficient at m^2 = 1: %.3e" % ((S, T, U), taylor(dM, lam, 0, 2).simplify_full(), dM.subs(m2=1, t=0).diff(lam, 3).subs(lam=0).n() / 6))
