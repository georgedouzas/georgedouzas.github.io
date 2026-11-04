# Exercise exr-ppde-dalembert-standing: d'Alembert's formula for phi = sin 2x, psi = 0 gives the standing wave
# sin 2x cos 2ct, and for phi = 0, psi = sin 2x it gives sin 2x sin(2ct)/(2c).
t, x, s, c = var("t x s c")
assume(c > 0)
def dalembert(phi, psi):
    return (phi.subs(x=x - c * t) + phi.subs(x=x + c * t)) / 2 + integrate(psi.subs(x=s), s, x - c * t, x + c * t) / (2 * c)
u1 = dalembert(sin(2 * x), SR(0))
print("phi = sin 2x, psi = 0: u =", u1.simplify_full(), "| equals sin 2x cos 2ct:", bool((u1 - sin(2 * x) * cos(2 * c * t)).simplify_full() == 0))
u2 = dalembert(SR(0), sin(2 * x))
print("phi = 0, psi = sin 2x: u =", u2.simplify_full(), "| equals sin 2x sin 2ct/(2c):", bool((u2 - sin(2 * x) * sin(2 * c * t) / (2 * c)).simplify_full() == 0))
print("both solve u_tt = c^2 u_xx:", all(bool((diff(u, t, 2) - c ** 2 * diff(u, x, 2)).simplify_full() == 0) for u in [u1, u2]))
