# Exercise exr-ren-lattice-dispersion: the lattice Laplacian 4/a^2 sin^2(ka/2) to order a^4, and its relative error
# along an axis and along the diagonal of a four-dimensional lattice.
var('k a')
print("series:", taylor(4 / a^2 * sin(k * a / 2)^2, a, 0, 4))
for ka in [0.1, 1.0]:
    axis = 1 - (sin(ka / 2) / (ka / 2))^2
    diag = 1 - (sin(ka / 4) / (ka / 4))^2
    print("ka = %.1f: relative error along an axis %.3e, along the diagonal %.3e" % (ka, axis, diag))
