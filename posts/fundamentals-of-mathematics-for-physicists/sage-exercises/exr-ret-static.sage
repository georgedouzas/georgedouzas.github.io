# Exercise exr-ret-static: the static Green's function of Box + m^2 from its Fourier representation
# (2 pi^2 r)^(-1) int_0^oo k sin(kr)/(k^2 + m^2) dk = e^(-mr)/(4 pi r), and the check (-Laplacian + m^2) G = 0 for r > 0.
k, r, m = var("k r m")
assume(m > 0, r > 0)
I = integrate(k * sin(k * r) / (k ** 2 + m ** 2), k, 0, oo)
print("radial Fourier integral:", I)
G0 = (I / (2 * pi ** 2 * r)).simplify_full()
print("G_0(r) =", G0, "| equals e^(-mr)/(4 pi r):", bool((G0 - exp(-m * r) / (4 * pi * r)).simplify_full() == 0))
radial_lap = lambda u: diff(u, r, 2) + 2 * diff(u, r) / r
print("(-Laplacian + m^2) G_0 = 0 for r > 0:", bool((-radial_lap(G0) + m ** 2 * G0).simplify_full() == 0))
print("massless limit is the Coulomb Green's function 1/(4 pi r):", bool(limit(G0, m=0) == 1 / (4 * pi * r)))
