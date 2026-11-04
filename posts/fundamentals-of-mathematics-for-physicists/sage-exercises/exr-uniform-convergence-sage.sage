# Exercise exr-uniform-convergence-sage: sup norms of f_n - f for f_n = x^n on [0, 1) (pointwise limit 0, sup stays 1),
# g_n = sin(nx)/n (sup 1/n, uniform) and h_n = x/(1 + n x^2) on R (sup 1/(2 sqrt n), uniform), computed exactly.
x, n = var("x n")
print("x^n on [0, 1): sup |x^n - 0| = lim_{x->1^-} x^n =", [limit(x ** k, x=1, dir="-") for k in [1, 5, 50]], "-> never below 1: not uniform")
print("sin(nx)/n: sup = 1/n for n = 1, 5, 50:", [QQ(1) / k for k in [1, 5, 50]], "-> 0: uniform")
hn = x / (1 + n * x ** 2)
crit = solve(diff(hn, x) == 0, x)
print("h_n = x/(1 + n x^2): critical points", [c.rhs() for c in crit], "; maximum value", hn.subs(x=1 / sqrt(n)).simplify_full(), "= 1/(2 sqrt n) -> 0: uniform")
print("integrals pass to the limit under uniform convergence: int_0^pi sin(nx)/n dx =", integrate(sin(n * x) / n, x, 0, pi), "-> 0 as n -> oo:", limit((1 - cos(n * pi)) / n ** 2, n=oo))
