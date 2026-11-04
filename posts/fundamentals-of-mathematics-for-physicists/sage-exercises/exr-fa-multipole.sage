# Exercise exr-fa-multipole: the multipole expansion of 1/|x - x'| in Legendre polynomials and in spherical harmonics,
# checked numerically for two points with l <= 3.
th, ph, th2, ph2 = var("theta phi theta2 phi2")
Y = lambda l, m, t, p: spherical_harmonic(l, m, t, p)
r1, r2 = 1, 3
n1 = (sin(0.4) * cos(1.1), sin(0.4) * sin(1.1), cos(0.4)); n2 = (sin(2.0) * cos(-0.7), sin(2.0) * sin(-0.7), cos(2.0))
cosg = sum(a * b for a, b in zip(n1, n2))
exact = (1 / sqrt(r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * cosg)).n(digits=12)
leg = sum(QQ(r1) ** l / QQ(r2) ** (l + 1) * legendre_P(l, cosg) for l in range(4)).n(digits=12)
sph = sum(4 * pi / (2 * l + 1) * QQ(r1) ** l / QQ(r2) ** (l + 1) * sum(Y(l, m, 0.4, 1.1) * conjugate(Y(l, m, 2.0, -0.7)) for m in range(-l, l + 1)) for l in range(4)).n(digits=12)
print("1/|x - x'| =", exact)
print("Legendre sum, l <= 3:", leg, "; spherical-harmonic sum, l <= 3:", sph.real(), "(imaginary part", abs(sph.imag()).n(digits=2), ")")
print("the two truncations agree:", bool(abs(leg - sph.real()) < 1e-10), "; error of the truncation", abs(exact - leg).n(digits=2), "; next term bound (r</r>)^4/r> =", (QQ(r1) ** 4 / QQ(r2) ** 5).n(digits=2))
t = var("t")
gen = taylor(1 / sqrt(1 - 2 * cosg * t + t ** 2), t, 0, 3)
print("generating function: coefficients of t^l are P_l(cos gamma) for l <= 3:", all(bool(abs(gen.coefficient(t, l) - legendre_P(l, cosg)) < 1e-12) for l in range(4)))
