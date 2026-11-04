# Exercise exr-spd-monopole-flux: the two potentials of the monopole of charge M on the unit sphere, their common field
# strength, their regularity at the poles and the flux (1/2pi) int da = M.
th, ph, M = var("th ph M")
aN = M / 2 * (1 - cos(th)); aS = -M / 2 * (1 + cos(th))      # phi components of a = a_phi dphi
print("da_N = (%s) dtheta ^ dphi, da_S = (%s) dtheta ^ dphi" % (diff(aN, th), diff(aS, th)))
print("a_N - a_S =", (aN - aS).simplify_full(), "(the gauge transformation exp(i M phi))")
print("a_N near theta = 0:", aN.taylor(th, 0, 2), "; a_S near theta = pi:", aS.subs(th=pi - var("u")).taylor(var("u"), 0, 2))
print("(1/2pi) int da =", (integrate(integrate(diff(aN, th), th, 0, pi), ph, 0, 2 * pi) / (2 * pi)).simplify_full())
