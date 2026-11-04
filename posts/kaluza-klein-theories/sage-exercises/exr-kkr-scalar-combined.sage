# Exercise exr-kkr-scalar-combined: both new terms at once. With g = eta, A_z = B v (F_{vz} = B, F^2 = 2 B^2) and
# phi = phi(x), the reduced curvature scalar is R5 = (1/4) phi^2 F^2 - (2/phi) Box phi = phi^2 B^2/2 + 2 phi''/phi.
M = Manifold(5, "M", structure="Lorentzian"); X = M.chart("t x v z y"); t, x, v, z, y = X[:]
B = var("B"); eta = [1, -1, -1, -1]; A = [0, 0, 0, B * v]; phi = function("phi")(x)
G = M.metric("G")
for i in range(4):
    for j in range(i, 4): G[i, j] = (eta[i] if i == j else 0) - phi^2 * A[i] * A[j]
    G[i, 4] = -phi^2 * A[i]
G[4, 4] = -phi^2
R5 = G.ricci_scalar().expr().simplify_full()
print("R5 =", R5)
print("[PASS] R5 = phi^2 B^2/2 + 2 phi''/phi = (1/4) phi^2 F^2 - (2/phi) Box phi:", bool((R5 - (phi^2 * B^2 / 2 + 2 * diff(phi, x, 2) / phi)).simplify_full() == 0))
