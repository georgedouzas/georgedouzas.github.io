# Exercise exr-ems-capacitance-ellipsoid (Macchi et al., Problem 2.14): capacitance of a prolate conducting ellipsoid.
a, b, h = var('a b h', domain='positive')
c = sqrt(a^2 - b^2)
C = 8 * pi * c / log((a + c) / (a - c))
print("limit b -> a:", limit(C, b=a))
for (hv, bv) in [(1.0, 0.01), (1.0, 0.001)]:
    exact = C.subs(a=hv / 2, b=bv).n()
    print("wire h = %s, b = %s: ellipsoid %.5f, 2 pi h/log(h/b) = %.5f" % (hv, bv, exact, (2 * pi * hv / log(hv / bv)).n()))
