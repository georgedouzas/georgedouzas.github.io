# Exercise exr-qst-bose-fermi-integrals: int_0^oo x^(s-1)/(e^x - 1) dx = Gamma(s) zeta(s) and
# int_0^oo x^(s-1)/(e^x + 1) dx = (1 - 2^(1-s)) Gamma(s) zeta(s), by term-by-term integration, for s = 2, 3, 4, 6, with
# a numerical cross-check.
x, n = var("x n"); assume(x > 0, n > 0)
for s in [2, 3, 4, 6]:
    term = integrate(x ** (s - 1) * exp(-n * x), x, 0, oo)
    B = sum(term, n, 1, oo); F = sum((-1) ** (n + 1) * term, n, 1, oo)
    qB = numerical_integral(x ** (s - 1) / (exp(x) - 1), 0, 100)[0]; qF = numerical_integral(x ** (s - 1) / (exp(x) + 1), 0, 100)[0]
    print("s = %d: Bose = %s (%.10f vs quadrature %.10f), Fermi = %s, ratio Fermi/Bose = %s" % (s, B, B.n(), qB, F, (F / B).simplify_full()))
