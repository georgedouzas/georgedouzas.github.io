# Exercise exr-grb-orbit-vs-static (Blennow-Ohlsson, Problems 2.99 and 2.100): a circular geodesic at radius r0 in the
# Schwarzschild metric (rs = 2GM, c = 1) and a static observer at the same radius: four-velocities, their relative
# speed, the ratio of their clocks between two meetings, and the numbers for a GPS orbit.
r0, rs = var('r0 rs'); assume(r0 > 3 * rs / 2, rs > 0)
f = 1 - rs / r0
Om = sqrt(rs / (2 * r0^3))                       # dphi/dt on a circular geodesic (Kepler's law, exact)
alpha = 1 / sqrt(f - r0^2 * Om^2)                # dt/dtau from g(U, U) = 1
beta = alpha * Om
print("alpha^2 = 1/(1 - 3 rs/(2 r0)):", bool((alpha^2 - 1 / (1 - 3 * rs / (2 * r0))).simplify_full() == 0))
print("normalisation f alpha^2 - r0^2 beta^2 = 1:", bool((f * alpha^2 - r0^2 * beta^2).simplify_full() == 1))
gam = (f * alpha * (1 / sqrt(f))).simplify_full()   # g(U, V) with V = f^(-1/2) d_t
v = sqrt(1 - 1 / gam^2).simplify_full()
print("relative speed v^2 =", (v^2).factor(), "  equals rs/(2 (r0 - rs)):", bool((v^2 - rs / (2 * (r0 - rs))).simplify_full() == 0))
ratio = (sqrt(1 - 3 * rs / (2 * r0)) / sqrt(f))
print("tau_orbit/tau_static = 1/gamma:", bool((ratio - 1 / gam).simplify_full() == 0))
num = {rs: 8.87e-3, r0: 2.656e7}
print("GPS: v = %.1f m/s, 1 - tau_orbit/tau_static = %.3e" % (v.subs(num) * 3e8, 1 - ratio.subs(num)))
print("GPS: one orbit takes %.2f h of coordinate time" % ((2 * pi / Om).subs(num) / 3e8 / 3600))
