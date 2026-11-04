# Exercise exr-ren-cas-zeta: zeta'(-4) = 3 zeta(5)/(4 pi^4) and the massless Casimir energy -pi^2 zeta'(-4)/l^4 at l = 1.
s = var('s')
zp = diff(zeta(s), s).subs(s=-4).n(60)
print("zeta'(-4) = %.12f, 3 zeta(5)/(4 pi^4) = %.12f" % (zp, (3 * zeta(5) / (4 * pi^4)).n(60)))
print("V(l = 1) = %.6f" % (-pi^2 * zp).n())
