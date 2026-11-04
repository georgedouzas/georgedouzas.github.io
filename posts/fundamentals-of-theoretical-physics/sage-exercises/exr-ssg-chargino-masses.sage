# Exercise exr-ssg-chargino-masses: the chargino mass matrix of the supersymmetric Standard Model (Nath eq. 10.43),
# M_C = [[M2, sqrt2 MW sin(beta)], [sqrt2 MW cos(beta), mu]], real parameters. The physical masses are the singular values,
# the square roots of the eigenvalues of M_C M_C^T; we check Nath's closed form (10.49) and evaluate a numerical case.
M2, mu, MW, b = var("M2 mu MW beta", domain="real")
MC = matrix(SR, [[M2, sqrt(2) * MW * sin(b)], [sqrt(2) * MW * cos(b), mu]])
S = MC * MC.transpose()
tr = S.trace().simplify_full(); dt = S.det().simplify_full()
print("tr(M M^T) =", tr, "   det(M M^T) =", dt.factor())
print("[PASS] tr = M2^2 + mu^2 + 2 MW^2:", bool((tr - (M2^2 + mu^2 + 2 * MW^2)).simplify_full() == 0))
print("[PASS] det = (M2 mu - MW^2 sin 2beta)^2:", bool((dt - (M2 * mu - MW^2 * sin(2 * b))^2).simplify_full() == 0))
pm = lambda e: (tr + e * sqrt(tr^2 - 4 * (M2 * mu - MW^2 * sin(2 * b))^2)) / 2
x = var("x")
cp = (x^2 - tr * x + dt)
print("[PASS] m^2 = 1/2[(M2^2 + mu^2 + 2MW^2) -+ sqrt((M2^2 + mu^2 + 2MW^2)^2 - 4(M2 mu - MW^2 sin 2beta)^2)] are the roots of the characteristic polynomial:",
      all(bool(cp.subs(x=pm(e)).simplify_full() == 0) for e in (1, -1)))
vals = {M2: 200, mu: 300, MW: 80.4, b: arctan(10)}
N = MC.subs(vals).change_ring(RDF)
sv = sorted(N.singular_values())
print("M2 = 200 GeV, mu = 300 GeV, tan(beta) = 10: chargino masses %.1f GeV and %.1f GeV" % tuple(sv))
print("[PASS] the singular values agree with the closed form:", all(abs(sv[i] - sqrt(pm(e).subs(vals)).n()) < 1e-6 for i, e in ((0, -1), (1, 1))))
