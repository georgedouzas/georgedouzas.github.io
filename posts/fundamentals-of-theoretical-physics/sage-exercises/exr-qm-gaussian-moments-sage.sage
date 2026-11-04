# Exercise exr-qm-gaussian-moments-sage: moments of psi = sqrt 2 pi^(-1/4) ell^(-3/2) x exp(-x^2/(2 ell^2)).
x = var("x", domain="real"); hbar, ell = var("hbar ell", domain="positive")
psi = sqrt(2) * pi ** (-1 / 4) * ell ** (-3 / 2) * x * exp(-x ** 2 / (2 * ell ** 2))
E = lambda g: integrate(g, x, -oo, oo).canonicalize_radical()
print("norm =", E(psi ** 2))
mx = E(x * psi ** 2); mp = E(psi * (-I * hbar) * diff(psi, x))
vx = E(x ** 2 * psi ** 2) - mx ** 2; vp = E(psi * (-hbar ** 2) * diff(psi, x, 2)) - mp ** 2
print("<x> =", mx, " <p> =", mp, " (Delta x)^2 =", vx, " (Delta p)^2 =", vp)
print("Delta x Delta p =", sqrt(vx * vp).canonicalize_radical())
