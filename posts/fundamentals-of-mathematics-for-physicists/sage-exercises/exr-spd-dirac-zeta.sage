# Exercise exr-spd-dirac-zeta: spectral zeta functions of D^2 on the unit S^2, 4 zeta(2s - 1), and on the antiperiodic
# circle of radius 1, 2 (2^{2s} - 1) zeta(2s), at s = 0 and s = -1/2.
s = var("s")
zS2 = 4 * zeta(2 * s - 1); zC = 2 * (2 ^ (2 * s) - 1) * zeta(2 * s)
print("check S^2 at s = 2: series", sum(4.0 / k ** 3 for k in range(1, 20000)), "closed form", zS2.subs(s=2).n(digits=8))
print("check circle at s = 1: series", sum(2.0 / (m + 0.5) ** 2 for m in range(0, 200000)), "closed form", zC.subs(s=1).n(digits=6))
for val in (0, -1/2):
    print("s = %s: zeta_S2 = %s, zeta_circle = %s" % (val, zS2.subs(s=val), zC.subs(s=val)))
