# Exercise exr-fyd-mandelstam: s, t, u for 2 -> 2 scattering of equal masses m in the centre-of-mass frame, and
# s + t + u = 4 m^2.
k, m, th = var("k m theta", domain="positive")
E = sqrt(k ** 2 + m ** 2)
eta = diagonal_matrix([1, -1, -1, -1])
dot = lambda a, b: a * eta * b
p1 = vector([E, 0, 0, k]); p2 = vector([E, 0, 0, -k])
p3 = vector([E, k * sin(th), 0, k * cos(th)]); p4 = vector([E, -k * sin(th), 0, -k * cos(th)])
s = dot(p1 + p2, p1 + p2).simplify_full(); t = dot(p1 - p3, p1 - p3).simplify_full(); u = dot(p1 - p4, p1 - p4).simplify_full()
print("s =", s, "  t =", t, "  u =", u)
print("s + t + u =", (s + t + u).simplify_full())
