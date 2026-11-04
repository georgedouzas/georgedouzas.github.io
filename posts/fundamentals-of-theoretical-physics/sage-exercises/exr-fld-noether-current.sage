# Exercise exr-fld-noether-current: the U(1) current j^mu = i(phibar d^mu phi - phi d^mu phibar) of a complex scalar in
# 1+1 dimensions, signature (+, -): d_mu j^mu = i(phibar Box phi - phi Box phibar), zero on solutions of the
# Klein-Gordon equation; checked for a superposition of two plane waves on the mass shell.
t, x, m = var("t x m", domain="real")
f = function("f")(t, x); g = function("g")(t, x)
phi, phib = f + I * g, f - I * g
box = lambda F: diff(F, t, 2) - diff(F, x, 2)
j0 = I * (phib * diff(phi, t) - phi * diff(phib, t))
j1 = I * (phib * (-diff(phi, x)) - phi * (-diff(phib, x)))          # d^1 = -d_1
div = diff(j0, t) + diff(j1, x)
print("d_mu j^mu = i(phibar Box phi - phi Box phibar):", bool((div - I * (phib * box(phi) - phi * box(phib))).simplify_full() == 0))
a, b, k1, k2 = var("a b k1 k2", domain="real")
w1, w2 = sqrt(k1**2 + m**2), sqrt(k2**2 + m**2)
psi = a * exp(-I * (w1 * t - k1 * x)) + b * exp(-I * (w2 * t - k2 * x)); psib = psi.conjugate()
J0 = I * (psib * diff(psi, t) - psi * diff(psib, t)); J1 = I * (-psib * diff(psi, x) + psi * diff(psib, x))
print("conserved for two on-shell plane waves:", bool((diff(J0, t) + diff(J1, x)).simplify_full() == 0))
print("charge density of one wave a e^{-i(wt - kx)}: j^0 =", (I * (psib * diff(psi, t) - psi * diff(psib, t))).subs(b=0).simplify_full())
