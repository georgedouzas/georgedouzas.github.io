# Exercise exr-ems-uniqueness-square: square with the side y = a at V0; the series at the centre.
def phi(xv, yv, N, V0=1.0, a=1.0):
    return sum(4 * V0 / (n * pi.n()) * sin(n * pi.n() * xv / a) * sinh(n * pi.n() * yv / a) / sinh(n * pi.n()) for n in range(1, N + 1, 2))
for N in [1, 3, 9, 41]:
    print("terms up to %d: phi(centre) = %.10f" % (N, phi(0.5, 0.5, N)))
print("boundary value at (0.5, 1) with 401 terms: %.4f" % phi(0.5, 1.0, 401))
