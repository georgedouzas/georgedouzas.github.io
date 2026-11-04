# Exercise exr-ems-energy-neumann: mutual inductance of two coaxial unit loops by Neumann's formula.
u = var('u')
def L12(d, a=1, b=1):
    return numerical_integral(a * b * cos(u) / (2 * sqrt(a^2 + b^2 + d^2 - 2 * a * b * cos(u))), 0, 2 * pi)[0]
for d in [1, 2, 5, 10]:
    est = pi.n() / (2 * d^3)
    print("d = %2d: Neumann %.6e, dipole estimate %.6e, ratio %.4f" % (d, L12(d), est, L12(d) / est))
