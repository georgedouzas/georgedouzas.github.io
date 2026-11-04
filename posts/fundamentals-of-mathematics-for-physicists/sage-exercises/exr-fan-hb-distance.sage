# Exercise exr-fan-hb-distance: distance from y = (1, 0) to Z = R(1, 1) in the sup norm, and the norming functional.
t = var('t')
dist = lambda tv: max(abs(1 - tv), abs(tv))
best = min((dist(k / 1000), k / 1000) for k in range(-2000, 2001))
print("min over t of |y - t(1,1)|_inf on a grid:", best)
phi = lambda v: (v[0] - v[1]) / 2
print("phi(x) = (x1 - x2)/2: phi(1,1) =", phi((1, 1)), " phi(y) =", phi((1, 0)), " norm |(1/2, -1/2)|_1 =", abs(QQ(1) / 2) + abs(-QQ(1) / 2))
