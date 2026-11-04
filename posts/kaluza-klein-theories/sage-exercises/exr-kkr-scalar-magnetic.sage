# Exercise exr-kkr-scalar-magnetic: a constant magnetic field. With g = eta, phi = 1 and A_z = B x (F_{xz} = B,
# a field along -v), F_{mu nu} F^{mu nu} = 2 B^2 > 0 and the reduced scalar is R5 = (1/4) F^2 = B^2/2, of the opposite
# sign to the electric case.
M = Manifold(5, "M", structure="Lorentzian"); X = M.chart("t x v z y"); t, x, v, z, y = X[:]
B = var("B"); eta = [1, -1, -1, -1]; A = [0, 0, 0, B * x]
G = M.metric("G")
for i in range(4):
    for j in range(i, 4): G[i, j] = (eta[i] if i == j else 0) - A[i] * A[j]
    G[i, 4] = -A[i]
G[4, 4] = -1
R5 = G.ricci_scalar().expr().simplify_full()
F2 = sum((diff(A[n], X[m]) - diff(A[m], X[n]))^2 * eta[m] * eta[n] for m in range(4) for n in range(4))
print("R5 =", R5, "; F^2 =", F2)
print("[PASS] R5 = (1/4) F^2 = B^2/2 for the constant magnetic field:", bool(R5 == F2 / 4) and bool(F2 == 2 * B^2))
