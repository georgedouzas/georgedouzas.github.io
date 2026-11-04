# Exercise exr-prg-massless-4d: the massless Euclidean propagator in four dimensions from proper time,
# int_0^oo ds (4 pi s)^{-2} e^{-x^2/(4 s)} = 1/(4 pi^2 x^2).
s = var("s"); x = var("x", domain="positive")
G = integrate((4 * pi * s) ** (-2) * exp(-x ** 2 / (4 * s)), s, 0, oo)
print("integral =", G, "  equals 1/(4 pi^2 x^2):", bool(G == 1 / (4 * pi ** 2 * x ** 2)))
