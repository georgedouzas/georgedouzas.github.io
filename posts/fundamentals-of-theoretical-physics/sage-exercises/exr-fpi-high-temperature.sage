# Exercise exr-fpi-high-temperature: expansion of G_beta(0) = coth(beta w/2)/(2 w) for small beta; the leading term
# is the zero Matsubara frequency (1/beta)(1/w^2).
b = var("b"); w = var("w", domain="positive")
G0 = coth(b * w / 2) / (2 * w)
print("G_beta(0) =", G0.series(b, 5))
print("zero-mode term (1/beta)/(w^2) =", 1 / (b * w ** 2))
