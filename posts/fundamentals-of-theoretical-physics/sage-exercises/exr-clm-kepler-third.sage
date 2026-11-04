# Exercise exr-clm-kepler-third: T = (area of the ellipse)/(areal velocity) = pi a b 2m/L with L^2 = m k a (1 - e^2).
a, e, m, k = var("a e m k", domain="positive")
assume(e < 1)
b = a * sqrt(1 - e**2); L = sqrt(m * k * a * (1 - e**2))
T = pi * a * b * 2 * m / L
print("T =", T.simplify_full(), "; T == 2 pi sqrt(m a^3/k):", bool((T**2 - 4 * pi**2 * m * a**3 / k).simplify_full() == 0))
