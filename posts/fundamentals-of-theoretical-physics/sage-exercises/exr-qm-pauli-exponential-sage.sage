# Exercise exr-qm-pauli-exponential-sage: exp(-i theta n.sigma/2) for n = (sin al, 0, cos al), and the induced rotation.
th, al = var("th al", domain="real")
sig = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
n = vector(SR, [sin(al), 0, cos(al)])
ns = sum(n[i] * sig[i] for i in range(3))
U = (-I * th * ns / 2).exp()
closed = cos(th / 2) * identity_matrix(SR, 2) - I * sin(th / 2) * ns
print("U - closed form =", (U - closed).apply_map(lambda e: e.rectform().trig_reduce().simplify_full()).list())
vals = {al: pi / 4, th: pi / 3}
Un = closed.subs(vals).simplify_full()
K = matrix(SR, [[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]]).subs(vals)
R = (identity_matrix(SR, 3) + sin(pi / 3) * K + (1 - cos(pi / 3)) * K * K).simplify_full()   # Rodrigues: rotation by theta about n
m = vector(SR, [1, 0, 0])
lhs = (Un.conjugate_transpose() * sig[0] * Un).simplify_full()
rhs = sum((R.inverse() * m)[i] * sig[i] for i in range(3)).simplify_full()
print("U^dagger sigma_1 U - (R^-1 e_1).sigma =", (lhs - rhs).simplify_full().list())
