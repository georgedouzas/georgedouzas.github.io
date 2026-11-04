# Exercise exr-qm-integrative-euclidean-oscillator: the Euclidean Mehler kernel acting on Hermite functions (units hbar = m = 1).
x, y = var("x y", domain="real"); w, tau = var("w tau", domain="positive")
KE = sqrt(w / (2 * pi * sinh(w * tau))) * exp(-w * ((x ** 2 + y ** 2) * cosh(w * tau) - 2 * x * y) / (2 * sinh(w * tau)))
psi = lambda n, X: (w / pi) ** (1 / 4) / sqrt(2 ** n * factorial(n)) * hermite(n, sqrt(w) * X) * exp(-w * X ** 2 / 2)
for n in range(3):
    out = integrate((KE * psi(n, y)).exponentialize(), y, -oo, oo)
    print("n =", n, ": int K_E psi_n / (e^(-(n + 1/2) w tau) psi_n(x)) =", (out / (exp(-(n + 1 / 2) * w * tau) * psi(n, x))).canonicalize_radical().simplify_full())
dk = (KE.subs(y=x) / (exp(-w * tau / 2) * psi(0, x) ** 2))
print("K_E(x, x; tau) / (e^(-w tau/2) psi_0(x)^2) at w = 1, x = 0.7, tau = 2, 5, 10:", [dk.subs(w=1, x=0.7, tau=s_).n(digits=8) for s_ in [2, 5, 10]])
print("-(1/tau) log K_E(0, 0; tau) at w = 1, tau = 20:", (-log(KE.subs(x=0, y=0, w=1, tau=20)) / 20).n(digits=6), "(-> E_0 = 1/2 as tau grows)")
