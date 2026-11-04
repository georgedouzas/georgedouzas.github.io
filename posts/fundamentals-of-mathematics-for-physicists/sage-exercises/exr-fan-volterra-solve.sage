# Exercise exr-fan-volterra-solve: u - 2 V u = 1 on [0, 1]: the resolvent formula, the Neumann series and a check.
x, t, n = var('x t n')
u = 1 + 2 * integrate(exp(2 * (x - t)), t, 0, x)
print("formula: u =", u.simplify_full())
print("Neumann series sum 2^n x^n/n! =", sum(2^n * x^n / factorial(n), n, 0, oo))
print("check u - 2 int_0^x u = 1:", bool((exp(2 * x) - 2 * integrate(exp(2 * t), t, 0, x)).simplify_full() == 1))
print("partial sums at x = 1 (N = 2, 5, 10) vs e^2:", [sum(2^k / factorial(k) for k in range(N)).n(digits=8) for N in (2, 5, 10)], exp(2).n(digits=8))
