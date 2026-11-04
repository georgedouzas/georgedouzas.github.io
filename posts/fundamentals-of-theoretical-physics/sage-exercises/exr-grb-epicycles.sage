# Exercise exr-grb-epicycles (Blennow-Ohlsson, Problem 2.75): small radial oscillations about a circular geodesic of
# the Schwarzschild metric (m = GM, c = 1). With (dr/dtau)^2 = E^2 - V(r), V = (1 - 2m/r)(1 + L^2/r^2), the radial
# frequency is omega_r^2 = V''(r0)/2; the ratio of the radial period to the orbital period is (1 - 6m/r0)^(-1/2).
r, r0, m, L = var('r r0 m L'); assume(r0 > 6 * m, m > 0)
V = (1 - 2 * m / r) * (1 + L^2 / r^2)
L2 = solve(diff(V, r).subs(r=r0) == 0, L^2)[0].rhs()
print("circular orbit: L^2 =", L2.factor())
wr2 = (diff(V, r, 2).subs(r=r0) / 2).subs(L^2 == L2).simplify_full()
wphi2 = (L2 / r0^4).simplify_full()               # (dphi/dtau)^2 on the circular orbit
ratio2 = (wphi2 / wr2).simplify_full()
print("(T_r/T_phi)^2 =", ratio2.factor(), "  equals 1/(1 - 6m/r0):", bool((ratio2 - 1 / (1 - 6 * m / r0)).simplify_full() == 0))
eps = var('eps')
adv = taylor(2 * pi * (sqrt(ratio2.subs(m=eps * r0)) - 1), eps, 0, 1)
print("perihelion advance per orbit to first order in m/r0:", adv.subs(eps=m / r0))
print("Mercury: 6 pi GM/(c^2 a) per orbit = %.3e rad (a = 5.79e10 m)" % (6 * pi.n() * 1476.6 / 5.79e10))
