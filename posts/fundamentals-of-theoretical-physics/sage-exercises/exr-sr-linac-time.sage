# Exercise exr-sr-linac-time (Blennow-Ohlsson, Problem 1.111): an electron accelerated from rest by 10^4 V/m over 100 m.
m = 0.511e6; eE = 1e4; L = 100.0; c = 2.99792458e8                 # eV, eV/m, m
gf = 1 + eE*L/m
var('E')
t_m = numerical_integral(E/(eE*sqrt(E^2 - m^2)), m, gf*m)[0]        # time in metres (c = 1)
print("final kinetic energy %.3f MeV, gamma_f = %.4f" % (eE*L/1e6, gf))
print("time = %.2f m/c = %.1f ns   (closed form m sqrt(gamma^2 - 1)/(eE) = %.2f m)" % (t_m, t_m/c*1e9, m*sqrt(gf^2 - 1)/eE))
