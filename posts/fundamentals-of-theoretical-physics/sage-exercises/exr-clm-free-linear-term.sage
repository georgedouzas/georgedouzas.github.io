# Exercise exr-clm-free-linear-term: S(eps) for the free particle along v t + eps t (T - t).
t, eps = var("t eps"); m, v, T = var("m v T", domain="positive")
x = v * t + eps * t * (T - t)
S = integrate(m / 2 * diff(x, t)**2, t, 0, T).expand()
print("S(eps) =", S.collect(eps))
print("linear coefficient:", S.coefficient(eps, 1), "; quadratic coefficient:", S.coefficient(eps, 2))
