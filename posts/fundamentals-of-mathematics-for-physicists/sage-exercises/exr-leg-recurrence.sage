# Exercise exr-leg-recurrence: the generating-function identity (1 - 2xt + t^2) dg/dt = (x - t) g and the three-term
# recurrence (n + 1) P_{n+1} = (2n + 1) x P_n - n P_{n-1} for n <= 6.
x, t = var("x t")
g = (1 - 2 * x * t + t ** 2) ** (-1 / 2)
print("(1 - 2xt + t^2) dg/dt - (x - t) g = 0:", bool(((1 - 2 * x * t + t ** 2) * diff(g, t) - (x - t) * g).simplify_full() == 0))
series = g.taylor(t, 0, 4)
print("coefficients of t^0..t^4 are P_0..P_4:", all(bool((series.coefficient(t, k) - legendre_P(k, x)).expand() == 0) for k in range(5)))
ok = all(bool(((k + 1) * legendre_P(k + 1, x) - (2 * k + 1) * x * legendre_P(k, x) + k * legendre_P(k - 1, x)).expand() == 0) for k in range(1, 7))
print("recurrence (n + 1) P_{n+1} = (2n + 1) x P_n - n P_{n-1} for n = 1..6:", ok)
