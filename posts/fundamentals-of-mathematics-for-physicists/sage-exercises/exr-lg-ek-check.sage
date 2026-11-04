# Exercise exr-lg-ek-check: Euler's kinematic equations. For R(t) = e^{phi L_3} e^{theta L_1} e^{psi L_3} with the Euler
# angles functions of t, R^{-1} dR/dt = omega.L (body) and dR/dt R^{-1} = Omega.L (space), with Omega = R omega.
t = var("t")
th, ph, ps = [function(f)(t) for f in ("theta", "phi", "psi")]
L = [matrix(SR, [[0, 0, 0], [0, 0, -1], [0, 1, 0]]), matrix(SR, [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]), matrix(SR, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])]
def rot(a, u): return identity_matrix(SR, 3) + sin(u) * L[a] + (1 - cos(u)) * L[a]^2
R = rot(2, ph) * rot(0, th) * rot(2, ps)
simp = lambda e: e.simplify_full().trig_reduce().simplify_full()
comp = lambda X: vector(SR, [simp(-(L[a] * X).trace() / 2) for a in range(3)])   # X = x_a L_a  ->  x_a
Rd = diff(R, t)
body, space = R.transpose() * Rd, Rd * R.transpose()
print("R^{-1} dR/dt and dR/dt R^{-1} antisymmetric:", bool((body + body.transpose()).apply_map(simp) == 0), bool((space + space.transpose()).apply_map(simp) == 0))
w, W = comp(body), comp(space)
dth, dph, dps = diff(th, t), diff(ph, t), diff(ps, t)
w_exp = vector(SR, [dth * cos(ps) + dph * sin(th) * sin(ps), -dth * sin(ps) + dph * sin(th) * cos(ps), dps + dph * cos(th)])
W_exp = vector(SR, [dth * cos(ph) + dps * sin(th) * sin(ph), dth * sin(ph) - dps * sin(th) * cos(ph), dph + dps * cos(th)])
print("body angular velocity omega =", w)
print("equals the left-invariant forms sigma^a(d/dt):", bool((w - w_exp).apply_map(simp) == 0))
print("space angular velocity Omega =", W)
print("equals the right-invariant forms:", bool((W - W_exp).apply_map(simp) == 0), "; Omega = R omega:", bool((W - R * w).apply_map(simp) == 0))
print("|omega|^2 = |Omega|^2 =", simp(w * w), ":", bool(simp(w * w - W * W) == 0))
