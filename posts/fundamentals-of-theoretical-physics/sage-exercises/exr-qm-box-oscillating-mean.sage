# Exercise exr-qm-box-oscillating-mean: <x>(t), <p>(t) for (phi_1 + phi_2)/sqrt 2 in the box, and Ehrenfest.
x, t = var("x t", domain="real"); hbar, m, L = var("hbar m L", domain="positive")
phi = lambda n: sqrt(2 / L) * sin(n * pi * x / L)
E = lambda n: n ** 2 * pi ** 2 * hbar ** 2 / (2 * m * L ** 2)
Psi = (phi(1) * exp(-I * E(1) * t / hbar) + phi(2) * exp(-I * E(2) * t / hbar)) / sqrt(2)
w = (E(2) - E(1)) / hbar
mx = integrate((x * Psi * Psi.conjugate()).expand(), x, 0, L)
mp = integrate((Psi.conjugate() * (-I * hbar) * diff(Psi, x)).expand(), x, 0, L)
print("<x>(t) - (L/2 - 16 L/(9 pi^2) cos(w t)) =", (mx - (L / 2 - 16 * L / (9 * pi ** 2) * cos(w * t))).exponentialize().simplify_full())
print("<p>(t) - (8 hbar/(3 L)) sin(w t) =", (mp - 8 * hbar / (3 * L) * sin(w * t)).exponentialize().simplify_full())
print("m d<x>/dt - <p> =", (m * diff(mx, t) - mp).exponentialize().simplify_full())
print("omega =", w)
