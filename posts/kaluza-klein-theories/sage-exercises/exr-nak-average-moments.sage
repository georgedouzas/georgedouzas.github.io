# Exercise exr-nak-average-moments: averages of products of the coordinates n_a over the unit sphere,
# <n_a n_b> = delta_ab / 3 and <n_a n_b n_c n_d> = (delta_ab delta_cd + delta_ac delta_bd + delta_ad delta_bc) / 15,
# by integration; the first gives <K_a . K_b> = delta_ab - delta_ab / 3 = (2/3) delta_ab.
th, ph = var("th ph")
n = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
avg = lambda f: integrate(integrate(f * sin(th), th, 0, pi), ph, 0, 2 * pi) / (4 * pi)
d = lambda a, b: 1 if a == b else 0
M2 = matrix(QQ, 3, 3, lambda a, b: avg(n[a] * n[b]))
print("<n_a n_b> =", M2.list())
print("[PASS] <n_a n_b> = delta_ab / 3:", M2 == identity_matrix(QQ, 3) / 3)
ok = all(bool(avg(n[a] * n[b] * n[c] * n[e]) == (d(a, b) * d(c, e) + d(a, c) * d(b, e) + d(a, e) * d(b, c)) / 15) for a in range(3) for b in range(3) for c in range(3) for e in range(3))
print("[PASS] <n_a n_b n_c n_d> = (delta delta + delta delta + delta delta) / 15:", ok)
print("[PASS] <K_a . K_b> = delta_ab - <n_a n_b> = (2/3) delta_ab:", identity_matrix(QQ, 3) - M2 == 2 / 3 * identity_matrix(QQ, 3))
