# Exercise exr-gr-lcdm-age: the age of a flat universe with Omega_m = 0.3, Omega_Lambda = 0.7, H0 t0 = int_0^1 da/(a H/H0)
# with H/H0 = sqrt(Omega_m a^{-3} + Omega_Lambda), numerically and in closed form; with H0 = 70 km/s/Mpc, in Gyr.
a = var('a'); Om, OL = 0.3, 0.7
num = numerical_integral(1/(a*sqrt(Om/a^3 + OL)), 0, 1)[0]
closed = (2/(3*sqrt(OL))*arcsinh(sqrt(OL/Om))).n()
print("H0 t0: numerical %.6f, closed form %.6f" % (num, closed))
H0 = 70*1e3/3.0857e22                        # in 1/s
Gyr = 3.15576e16
print("1/H0 = %.2f Gyr; t0 = %.2f Gyr; matter-only age 2/(3 H0) = %.2f Gyr" % (1/H0/Gyr, num/H0/Gyr, 2/3/H0/Gyr))
