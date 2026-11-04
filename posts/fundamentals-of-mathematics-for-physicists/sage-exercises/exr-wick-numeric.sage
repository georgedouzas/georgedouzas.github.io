# Exercise exr-wick-numeric: the Minkowski integral of 1/(k^2 - 4 + i eps) numerically for eps = 1e-4, against
# -i pi/2, and the exact Euclidean integral -i times the integral of 1/(k^2 + 4).
eps = 1e-4
re_ = numerical_integral(lambda k: (1/(k^2 - 4 + I*eps)).real(), -500, 500)[0]
im_ = numerical_integral(lambda k: (1/(k^2 - 4 + I*eps)).imag(), -500, 500)[0]
print("numerical Minkowski integral:", CC(re_, im_), "   expected -i pi/2 =", (-I*pi/2).n())
print("imaginary part agrees to 1e-3:", abs(im_ + pi.n()/2) < 1e-3, "  real part small:", abs(re_) < 1e-2)
k = var("k")
print("Euclidean: -i * integral of 1/(k^2 + 4) =", -I*integrate(1/(k^2 + 4), k, -oo, oo))
