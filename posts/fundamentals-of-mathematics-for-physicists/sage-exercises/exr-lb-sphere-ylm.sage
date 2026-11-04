# Exercise exr-lb-sphere-ylm: the spherical harmonics as eigenfunctions of the Laplacian on the unit sphere, -Delta Y_lm =
# l(l + 1) Y_lm for l <= 3, their orthonormality for l, l' <= 2, and the count 2l + 1 of eigenfunctions per eigenvalue.
th, ph = var("theta phi", domain="real")
lap = lambda f: (diff(sin(th) * diff(f, th), th) / sin(th) + diff(f, ph, 2) / sin(th) ** 2)
Y = lambda l, m: spherical_harmonic(l, m, th, ph)
for l in range(4):
    print("l = %d: -Delta Y_lm = %d Y_lm for all m:" % (l, l * (l + 1)), all(bool((lap(Y(l, m)) + l * (l + 1) * Y(l, m)).simplify_full() == 0) for m in range(-l, l + 1)), "; number of eigenfunctions:", 2 * l + 1)
ip = lambda f, g: integrate(integrate(conjugate(f) * g * sin(th), th, 0, pi), ph, 0, 2 * pi)
lm = [(l, m) for l in range(3) for m in range(-l, l + 1)]
G = matrix(SR, len(lm), len(lm), lambda a, b: ip(Y(*lm[a]), Y(*lm[b])).simplify_full())
print("Gram matrix of the nine Y_lm with l <= 2 is the identity:", G == identity_matrix(len(lm)))
print("total number of eigenfunctions with l <= 3:", sum(2 * l + 1 for l in range(4)))
