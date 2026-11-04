# Exercise exr-nak-consistency-l2-component: the part of the Killing inner product that the truncation cannot absorb.
# K_1 . K_1 = 2/3 + q with q = 1/3 - n_1^2, a pure l = 2 harmonic: q is orthogonal to the constants (the l = 0 zero
# mode kept by the ansatz) and to the l = 1 harmonics n_a, and its mean square 4/45 is the variance of K_1 . K_1.
th, ph = var("th ph")
n = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
avg = lambda f: integrate(integrate(f * sin(th), th, 0, pi), ph, 0, 2 * pi) / (4 * pi)
q = 1 / 3 - n[0]^2
print("[PASS] q is orthogonal to the constants and to n_1, n_2, n_3:", bool(avg(q) == 0) and all(bool(avg(q * n[a]) == 0) for a in range(3)))
print("[PASS] <q^2> = 4/45, the variance of K_1 . K_1:", bool(avg(q^2) == 4 / 45))
# the l = 2 source for a field strength along a = 1: (1/4) q F^1 F^1 with F^1 F^1 = -2 c^2 for A^1_x = c t
c = var("c")
print("l = 2 part of R_6 for A^1_x = c t:", (q * (-2 * c^2) / 4).simplify_full())
