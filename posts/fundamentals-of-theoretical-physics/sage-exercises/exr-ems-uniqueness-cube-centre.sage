# Exercise exr-ems-uniqueness-cube-centre (Pierrus, Question 3.17): box with the lid at V0; the series at the centre of a cube.
def phi_centre(N, V0=1.0, a=1.0):
    tot = 0
    for m in range(1, N + 1, 2):
        for n in range(1, N + 1, 2):
            g = pi.n() * sqrt(m^2 + n^2) / a
            tot += 16 * V0 / (pi.n()^2 * m * n) * sin(m * pi.n() / 2) * sin(n * pi.n() / 2) * sinh(g * a / 2) / sinh(g * a)
    return tot
for N in [1, 5, 21, 81]:
    print("terms up to %d: phi(centre) = %.8f" % (N, phi_centre(N)))
print("1/6 = %.8f" % (1 / 6))
