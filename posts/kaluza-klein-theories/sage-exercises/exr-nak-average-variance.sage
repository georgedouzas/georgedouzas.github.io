# Exercise exr-nak-average-variance: how far the Killing inner products are from their average. Over the unit sphere
# the variance of K_1 . K_1 = 1 - n_1^2 is <(K_1.K_1)^2> - <K_1.K_1>^2 = 4/45, and the off-diagonal K_1 . K_2 = -n_1 n_2
# has mean 0 and mean square 1/15: the pointwise coupling fluctuates by 1/sqrt(5), about 45 per cent, of its mean.
th, ph = var("th ph")
n = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
avg = lambda f: integrate(integrate(f * sin(th), th, 0, pi), ph, 0, 2 * pi) / (4 * pi)
K11 = 1 - n[0]^2; K12 = -n[0] * n[1]
var11 = avg(K11^2) - avg(K11)^2
print("<K_1.K_1> =", avg(K11), "  variance =", var11, "  relative spread =", sqrt(var11) / avg(K11))
print("[PASS] the variance of K_1 . K_1 over the sphere is 4/45:", bool(var11 == 4 / 45))
print("[PASS] <K_1 . K_2> = 0 and <(K_1 . K_2)^2> = 1/15:", bool(avg(K12) == 0) and bool(avg(K12^2) == 1 / 15))
