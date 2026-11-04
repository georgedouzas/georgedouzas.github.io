# Exercise exr-nak-harmonic-l2: the spherical harmonics behind the consistency problem. On the unit sphere the
# Laplacian Delta f = (1/sin th) d_th (sin th d_th f) + (1/sin^2 th) d_ph^2 f has Delta n_a = -2 n_a (l = 1, the
# Killing potentials) and Delta (n_a n_b - delta_ab/3) = -6 (n_a n_b - delta_ab/3) (l = 2); hence
# K_a . K_b = (2/3) delta_ab - (n_a n_b - delta_ab/3) splits into an l = 0 and an l = 2 part.
th, ph = var("th ph")
n = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
Lap = lambda f: (diff(sin(th) * diff(f, th), th) / sin(th) + diff(f, ph, 2) / sin(th)^2).simplify_full()
print("[PASS] Delta n_a = -2 n_a (l = 1, eigenvalue -l(l+1)):", all(bool((Lap(n[a]) + 2 * n[a]).simplify_full() == 0) for a in range(3)))
d = lambda a, b: 1 if a == b else 0
Q = lambda a, b: n[a] * n[b] - d(a, b) / 3
print("[PASS] Delta (n_a n_b - delta_ab/3) = -6 (n_a n_b - delta_ab/3) (l = 2):", all(bool((Lap(Q(a, b)) + 6 * Q(a, b)).simplify_full() == 0) for a in range(3) for b in range(3)))
K = [vector([-sin(ph), -cot(th) * cos(ph)]), vector([cos(ph), -cot(th) * sin(ph)]), vector([0, 1])]
gam = diagonal_matrix(SR, [1, sin(th)^2])
print("[PASS] K_a . K_b = (2/3) delta_ab - (n_a n_b - delta_ab/3):", all(bool((K[a] * gam * K[b] - (2 * d(a, b) / 3 - Q(a, b))).simplify_full() == 0) for a in range(3) for b in range(3)))
