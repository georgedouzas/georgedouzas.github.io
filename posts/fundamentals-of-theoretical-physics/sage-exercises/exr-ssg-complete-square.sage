# Exercise exr-ssg-complete-square: the identity F^* F + w F + w^* F^* = |F + w^*|^2 - |w|^2 in real components
# (F = a + i b, w = u + i v), the stationary point F = -w^*, and its nature: in the real variables the Hessian of
# F^* F is 2 times the identity, so the stationary point is a minimum of L_aux, whose value -|w|^2 enters
# the potential with the opposite sign, V = |w|^2.
a, b, u, v = var("a b u v", domain="real")
F = a + I * b; w = u + I * v
lhs = (F.conjugate() * F + w * F + w.conjugate() * F.conjugate()).expand()
rhs = ((F + w.conjugate()) * (F + w.conjugate()).conjugate() - w * w.conjugate()).expand()
print("[PASS] F^*F + wF + w^*F^* = |F + w^*|^2 - |w|^2:", bool((lhs - rhs).expand() == 0))
print("[PASS] L_aux is real:", bool(lhs.imag_part().simplify_full() == 0))
st = solve([diff(lhs, a) == 0, diff(lhs, b) == 0], [a, b], solution_dict=True)[0]
print("stationary point:", st)
print("[PASS] the stationary point is F = -w^* (a = -u, b = v):", bool(st[a] == -u) and bool(st[b] == v))
H = matrix([[diff(lhs, a, a), diff(lhs, a, b)], [diff(lhs, b, a), diff(lhs, b, b)]])
print("[PASS] the Hessian is 2 * identity: a minimum of L_aux with value -|w|^2:", H == 2 * identity_matrix(2) and bool(lhs.subs(st).expand() == -(u^2 + v^2)))
