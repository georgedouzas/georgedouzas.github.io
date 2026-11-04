# Exercise exr-fpi-zero-dim-moments: moments of the zero-dimensional Gaussian field theory from Z(J) = e^{J^2/(2 m^2)}:
# <phi^{2n}> = (2n - 1)!!/m^{2n}, odd moments vanish, and the direct integrals agree.
J, phi = var("J phi"); m = var("m", domain="positive")
Z = exp(J ** 2 / (2 * m ** 2))
mom = [diff(Z, J, k).subs(J=0) for k in range(11)]
print("derivatives at J = 0:", mom)
print("even moments = (2n-1)!!/m^{2n}:", all(bool(mom[2 * n] == prod(range(2 * n - 1, 0, -2)) / m ** (2 * n)) for n in range(6)), "  odd vanish:", all(mom[k] == 0 for k in range(1, 11, 2)))
norm = integrate(exp(-m ** 2 * phi ** 2 / 2), phi, -oo, oo)
print("direct integrals agree:", all(bool(integrate(phi ** k * exp(-m ** 2 * phi ** 2 / 2), phi, -oo, oo) / norm == mom[k]) for k in range(11)))
