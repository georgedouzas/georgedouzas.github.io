# Exercise exr-cfb-atom-collapse (Maggiore, Problem 10.3): a classical electron on a circular orbit around a proton loses
# energy by Larmor radiation; with E = -e^2/(8 pi eps0 r) and P = e^2 a^2/(6 pi eps0 c^3), a = e^2/(4 pi eps0 m r^2),
# the radius obeys dr/dt = -4 r_e^2 c/(3 r^2) and reaches 0 after tau = a0^3/(4 r_e^2 c).
r, t, re, c, a0, K, m = var('r t re c a0 K m'); assume(r > 0, re > 0, c > 0, a0 > 0, K > 0, m > 0)
# K = e^2/(4 pi eps0); classical electron radius re = K/(m c^2)
E = -K / (2 * r)
acc = K / (m * r^2)
P = 2 * K * acc^2 / (3 * c^3)                            # e^2 a^2/(6 pi eps0 c^3) = (2/3) K a^2/c^3
drdt = (-P / diff(E, r)).simplify_full()
print("dr/dt =", drdt, "  = -4 re^2 c/(3 r^2):", bool((drdt.subs(K=re * m * c^2) + 4 * re^2 * c / (3 * r^2)).simplify_full() == 0))
tau = integrate(3 * r^2 / (4 * re^2 * c), r, 0, a0)
print("collapse time tau =", tau)
print("numbers: tau = %.3e s (a0 = 5.29e-11 m, re = 2.82e-15 m)" % tau.subs(a0=5.29e-11, re=2.82e-15, c=2.998e8))
print("orbital period at a0: %.3e s, so about %.1e revolutions" % ((2 * pi * sqrt(m * a0^3 / K)).subs(m=9.109e-31, a0=5.29e-11, K=2.307e-28),
      (tau / (2 * pi * sqrt(m * a0^3 / K))).subs(m=9.109e-31, a0=5.29e-11, K=2.307e-28, re=2.82e-15, c=2.998e8)))
