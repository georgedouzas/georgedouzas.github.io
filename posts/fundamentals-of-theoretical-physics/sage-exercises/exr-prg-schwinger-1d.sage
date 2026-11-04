# Exercise exr-prg-schwinger-1d: the proper-time representation in one dimension,
# int_0^oo ds (4 pi s)^{-1/2} exp(-tau^2/(4 s) - w^2 s) = e^{-w |tau|}/(2 w).
s = var("s"); tau, w = var("tau w", domain="positive")
G = integrate((4 * pi * s) ** (-QQ(1) / 2) * exp(-tau ** 2 / (4 * s) - w ** 2 * s), s, 0, oo)
print("integral =", G.simplify_full())
print("equals e^{-w tau}/(2 w):", bool(G == exp(-w * tau) / (2 * w)))
