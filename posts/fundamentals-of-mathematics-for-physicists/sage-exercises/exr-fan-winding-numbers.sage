# Exercise exr-fan-winding-numbers: winding number of phi(z) = z^2 (z - 1/2)/(z - 3) about 0 along the unit circle,
# by the argument principle and by numerical integration of phi'/phi.
z = var("z")
phi = z ^ 2 * (z - 1/2) / (z - 3)
dlog = (diff(phi, z) / phi).partial_fraction(z)
print("phi'/phi =", dlog)
t = var("t")
integrand = (dlog.subs(z=exp(I * t)) * I * exp(I * t))
re = numerical_integral(real_part(integrand), 0, 2 * pi)[0]
im = numerical_integral(imag_part(integrand), 0, 2 * pi)[0]
print("(1/(2 pi i)) * contour integral =", round(im / (2 * pi.n()), 10), "+ i *", round(-re / (2 * pi.n()), 10))
print("zeros inside: 0 (double), 1/2; pole 3 outside: winding number 3, index of T_phi = -3")
