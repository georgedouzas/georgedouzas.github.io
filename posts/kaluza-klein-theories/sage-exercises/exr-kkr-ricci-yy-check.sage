# Exercise exr-kkr-ricci-yy-check: R_yy = phi Box phi + (phi^4/4) F^2 on two configurations with flat spacetime part
# (signature (+,-,-,-,-), Box = d_t^2 - nabla^2): the dilaton phi(t, x) with A = 0, and the constant electric field
# A_x = B t with phi = 1.
M = Manifold(5, "M", structure="Lorentzian"); X = M.chart("t x v z y"); t, x, v, z, y = X[:]
phi = function("phi")(t, x)
G = M.metric("G")
for i, e in enumerate([1, -1, -1, -1]): G[i, i] = e
G[4, 4] = -phi^2
Ryy = G.ricci()[4, 4].expr()
box = diff(phi, t, 2) - diff(phi, x, 2)
print("R_yy for eta - phi(t,x)^2 dy^2:", Ryy.simplify_full())
print("[PASS] R_yy = phi Box phi for A = 0:", bool((Ryy - phi * box).simplify_full() == 0))
B = var("B"); H = M.metric("H"); A = [0, B * t, 0, 0]; eta = [1, -1, -1, -1]
for i in range(4):
    for j in range(i, 4): H[i, j] = (eta[i] if i == j else 0) - A[i] * A[j]
    H[i, 4] = -A[i]
H[4, 4] = -1
R2 = H.ricci()[4, 4].expr().simplify_full()
print("R_yy for the electric field:", R2)
print("[PASS] R_yy = (1/4) F^2 = -B^2/2 for phi = 1, F_{tx} = B:", bool(R2 == -B^2 / 2))
