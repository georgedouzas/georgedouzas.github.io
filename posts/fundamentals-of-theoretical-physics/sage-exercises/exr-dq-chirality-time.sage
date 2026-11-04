# Exercise exr-dq-chirality-time: psi(t) = exp(-iHt)(1, 0, 0, 0) for H = p alpha_3 + m beta (Weyl representation) has
# <gamma_5>(t) = -1 + (m^2/E^2)(1 - cos 2Et); and [H, gamma_5] = 2 m beta gamma_5.
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
Z = zero_matrix(SR, 2); one = identity_matrix(SR, 2)
blk = lambda a, b, c, d: block_matrix([[a, b], [c, d]], subdivide=False)
g = [blk(Z, one, one, Z)] + [blk(Z, sk, -sk, Z) for sk in s]   # Weyl representation
g5 = I * g[0] * g[1] * g[2] * g[3]; I4 = identity_matrix(SR, 4)
m, p, t = var("m p t", domain="positive"); E = sqrt(p^2 + m^2)
H = p * g[0] * g[3] + m * g[0]
print("H^2 = E^2:", (H^2 - E^2 * I4).simplify_full() == 0)
Ut = cos(E * t) * I4 - I * sin(E * t) * H / E
psi = Ut * vector(SR, [1, 0, 0, 0])
ch = (psi.conjugate() * g5 * psi).simplify_full()
print("<gamma_5>(t) =", ch)
print("= -1 + (m^2/E^2)(1 - cos 2Et):", bool((ch - (-1 + m^2 / E^2 * (1 - cos(2 * E * t)))).simplify_full().trig_reduce() == 0))
print("[H, gamma_5] = 2 m beta gamma_5:", (H * g5 - g5 * H - 2 * m * g[0] * g5).simplify_full() == 0)
