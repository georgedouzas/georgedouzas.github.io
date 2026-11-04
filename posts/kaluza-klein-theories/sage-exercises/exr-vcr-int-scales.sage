# Exercise exr-vcr-int-scales: no separation of scales in AdS_4 x S^7. The restriction of the harmonic polynomial
# x1 x2 of R^8 to the round S^7 of radius a is an eigenfunction of the Laplacian with eigenvalue l(l+6)/a^2 = 16/a^2
# (l = 2). With a = 1/m and L = 1/(2m) the ratio of this Kaluza-Klein scale to the AdS curvature scale 1/L^2 is a pure
# number, independent of the flux m.
a, m = var("a m", domain="positive")
S = Manifold(7, "S7"); X = S.chart(" ".join("c%d" % k for k in range(7))); c = X[:]
gam = S.metric("gam"); w = 1
for k in range(7):
    gam[k, k] = a ^ 2 * w; w = w * sin(c[k]) ^ 2
# embedding coordinates: x1 = a cos c0, x2 = a sin c0 cos c1, ...
x1 = a * cos(c[0]); x2 = a * sin(c[0]) * cos(c[1])
f = S.scalar_field(x1 * x2 / a ^ 2)

lap = f.laplacian(gam).expr().simplify_full()
ratio = (lap / f.expr()).simplify_full()
print("Delta f / f =", ratio, "; equals -l(l+6)/a^2 with l = 2:", bool(ratio == -16 / a ^ 2))
print("KK mass scale 16/a^2 against AdS scale 1/L^2 with a = 1/m, L = 1/(2m):", (16 * m ^ 2) / (4 * m ^ 2))
