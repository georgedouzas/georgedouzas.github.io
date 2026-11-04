# Exercise exr-qm-mehler-trace-sage: trace of the Euclidean Mehler kernel and its heat equation.
x, y = var("x y", domain="real"); hbar, m, w, tau = var("hbar m w tau", domain="positive")
KE = sqrt(m * w / (2 * pi * hbar * sinh(w * tau))) * exp(-m * w * ((x ** 2 + y ** 2) * cosh(w * tau) - 2 * x * y) / (2 * hbar * sinh(w * tau)))
assume(cosh(w * tau) - 1 > 0)
tr = integrate(KE.subs(y=x), x, -oo, oo)
print("trace =", tr.simplify_full())
print("trace * 2 sinh(w tau/2) =", (tr * 2 * sinh(w * tau / 2)).simplify_full().exponentialize().canonicalize_radical())
res = (-hbar * diff(KE, tau) - (-hbar ** 2 / (2 * m) * diff(KE, x, 2) + m * w ** 2 * x ** 2 / 2 * KE)) / KE
print("(-hbar K_tau - H K)/K =", res.simplify_full().exponentialize().simplify_full())
