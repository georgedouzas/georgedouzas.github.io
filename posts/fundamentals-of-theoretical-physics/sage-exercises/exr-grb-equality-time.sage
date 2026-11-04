# Exercise exr-grb-equality-time (Blennow-Ohlsson, Problem 2.147): a flat universe of matter and radiation with
# Omega_r = x, Omega_m = 1 - x today; time elapsed since matter-radiation equality, a_eq = x/(1 - x), from
# t(a) = int_0^a da'/(a' H(a')) with H = H0 sqrt(Omega_r a^-4 + Omega_m a^-3).
a, x, H0, s = var('a x H0 s'); assume(x > 0, x < 1, a > 0)
Om = 1 - x
F = integrate(s / sqrt(x + Om * s), s, 0, a)        # H0 t(a)
F = F.simplify_full()
closed = 2 / (3 * Om^2) * ((Om * a - 2 * x) * sqrt(x + Om * a) + 2 * x^(3 / 2))
print("H0 t(a) = 2((Om a - 2x) sqrt(x + Om a) + 2 x^(3/2))/(3 Om^2):", bool((F - closed).simplify_full() == 0))
aeq = x / Om
dt = (closed.subs(a=1) - closed.subs(a=aeq)).simplify_full()
print("H0 (t0 - t_eq) =", dt)
for xv in (1e-3, 1e-4, 8.5e-5):
    print("x = %.1e: H0 (t0 - t_eq) = %.6f, H0 t_eq = %.3e" % (xv, dt.subs(x=xv).n(), closed.subs(a=aeq).subs(x=xv).n()))
