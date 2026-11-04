# Exercise exr-gth-so3-partial-masses: gauging the SO(3) vector model (a real triplet with kinetic term (D phi)^2/2,
# D_mu phi = d_mu phi + g A^a_mu L_a phi) at the vacuum phi0 = (0, 0, a): the mass matrix g^2 (L_a phi0).(L_b phi0) has
# eigenvalues g^2 a^2, g^2 a^2, 0; the Higgs mass is 2 lam a^2; the degrees of freedom are conserved.
g, a, lam = var("g a lam", domain="positive")
eps = lambda i, j, k: (i - j) * (j - k) * (k - i) / 2
L = [matrix(SR, 3, 3, lambda b, c: -eps(i, b, c)) for i in range(3)]
print("[PASS] [L_1, L_2] = L_3 (so(3) with f = eps):", L[0] * L[1] - L[1] * L[0] == L[2])
v0 = vector(SR, [0, 0, a])
M2 = matrix(SR, 3, 3, lambda i, j: g^2 * (L[i] * v0) * (L[j] * v0))
print("mass matrix:"); print(M2)
print("[PASS] eigenvalues g^2 a^2, g^2 a^2, 0 (the gauge boson of the unbroken SO(2) stays massless):", sorted(M2.eigenvalues()) == sorted([g^2 * a^2, g^2 * a^2, 0]))
p = vector(SR, var("p1 p2 p3")); V = lam / 4 * (p * p - a^2)^2
H = matrix(SR, 3, 3, lambda i, j: diff(V, p[i], p[j])).subs(dict(zip(p, v0)))
print("[PASS] the Higgs (radial) mass^2 is 2 lam a^2:", bool(H[2, 2] == 2 * lam * a^2))
print("degrees of freedom: before 3 + 3*2 =", 3 + 3 * 2, "; after 1 (Higgs) + 2*3 (massive) + 1*2 (massless) =", 1 + 2 * 3 + 2)
