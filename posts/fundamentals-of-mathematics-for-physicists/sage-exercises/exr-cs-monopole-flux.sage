# Exercise exr-cs-monopole-flux: the flux of the R-connection of the sphere. With e^3 = cos(theta) dphi and
# de^3 = -sin(theta) dtheta ^ dphi: the total flux is -4 pi; along the parallel theta = theta_0 the line integral is
# 2 pi cos(theta_0) while the flux through the cap above it is -2 pi (1 - cos(theta_0)): Stokes fails by 2 pi because
# e^3 is singular at the pole, and the gauge-transformed potential (cos(theta) - 1) dphi repairs it.
th, ph, th0 = var("th ph th0", domain="real")
assume(th0 > 0, th0 < pi)
F = -sin(th)                                                    # de^3 = F dtheta ^ dphi
print("total flux  int_S2 de^3 =", integrate(integrate(F, th, 0, pi), ph, 0, 2 * pi))
line = integrate(cos(th0), ph, 0, 2 * pi); cap = integrate(integrate(F, th, 0, th0), ph, 0, 2 * pi)
print("line integral of e^3 along theta = theta_0:", line)
print("flux through the cap theta < theta_0:", cap.simplify_full())
print("line - cap =", (line - cap).simplify_full(), " (a Dirac string at the north pole)")
A_N = cos(th) - 1                                               # e^3 - dphi: regular at theta = 0
print("with e^3 - dphi: line integral =", integrate(A_N.subs(th=th0), ph, 0, 2 * pi).simplify_full(),
      " equals the cap flux:", bool((integrate(A_N.subs(th=th0), ph, 0, 2 * pi) - cap).simplify_full() == 0))
print("and d(e^3 - dphi) = de^3:", bool(diff(A_N, th) == F))
