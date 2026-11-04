# Exercise exr-ems-image-sphere-check: the image -qR/d at R^2/d makes the grounded sphere an equipotential.
q, R, d, g = var('q R d gamma'); assume(R > 0, d > R)
r1sq = R^2 + d^2 - 2 * R * d * cos(g); r2sq = R^2 + (R^2 / d)^2 - 2 * R * (R^2 / d) * cos(g)
print("on the sphere |x - d'|^2 = (R/d)^2 |x - d|^2:", bool((r2sq - (R / d)^2 * r1sq).simplify_full() == 0), "-> q/|x-d| - (qR/d)/|x-d'| = 0")
r, th = var('r th'); assume(r > 0)
phi = q / (4 * pi * sqrt(r^2 + d^2 - 2 * r * d * cos(th))) - q * R / d / (4 * pi * sqrt(r^2 + R^4 / d^2 - 2 * r * R^2 / d * cos(th)))
sigma = (-diff(phi, r)).subs(r=R)
sigma_expected = -q * (d^2 - R^2) / (4 * pi * R * (R^2 + d^2 - 2 * R * d * cos(th))^(3/2))
vals = {q: 1, R: 1, d: 2}
print("sigma - expected at five angles (R = 1, d = 2):", [((sigma - sigma_expected).subs(vals).subs(th=tt)).n(digits=6) for tt in [0, 0.5, 1.2, 2.0, pi]])
tot = numerical_integral((2 * pi * R^2 * sigma_expected * sin(th)).subs(vals), 0, pi)[0]
print("total induced charge for R = 1, d = 2: %.10f (expected -qR/d = -0.5)" % tot)
