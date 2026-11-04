# Exercise exr-l2-sphere-norm: norms and inner products in L^2(S^2) with the area element sin(theta) dtheta dphi, the
# normalisation of the l = 1 spherical harmonics and their Gram matrix.
th, ph = var("theta phi", domain="real")
ip = lambda f, g: integrate(integrate(conjugate(f) * g * sin(th), th, 0, pi), ph, 0, 2 * pi)
f1, f2 = cos(th), sin(th) * exp(I * ph)
print("||cos theta||^2 =", ip(f1, f1), "; ||sin theta e^{i phi}||^2 =", ip(f2, f2), "; <cos theta, sin theta e^{i phi}> =", ip(f1, f2))
Y = [sqrt(3 / (8 * pi)) * sin(th) * exp(-I * ph), sqrt(3 / (4 * pi)) * cos(th), -sqrt(3 / (8 * pi)) * sin(th) * exp(I * ph)]
G = matrix(SR, 3, 3, lambda a, b: ip(Y[a], Y[b]).simplify_full())
print("Gram matrix of Y_{1,-1}, Y_{10}, Y_{11}:"); print(G)
print("agrees with Sage's spherical_harmonic:", all(bool((Y[m + 1] - spherical_harmonic(1, m, th, ph)).simplify_full() == 0) for m in [-1, 0, 1]))
