# Exercise exr-group-matrix-exponential: the flow of a linear vector field X(x) = A x is the matrix exponential
# Phi_t = exp(tA); checked for the rotation, the shear and the boost generator by solving dx/dt = A x with desolve.
t, s = var("t s", domain="real")
x0, y0 = var("x0 y0", domain="real")
x = function("x")(t); y = function("y")(t)
for name, A in [("rotation", matrix(QQ, [[0, -1], [1, 0]])), ("shear", matrix(QQ, [[0, 1], [0, 0]])), ("boost", matrix(QQ, [[0, 1], [1, 0]]))]:
    sol = desolve_system([diff(x, t) == A[0, 0] * x + A[0, 1] * y, diff(y, t) == A[1, 0] * x + A[1, 1] * y], [x, y], ics=[0, x0, y0], ivar=t)
    flow = vector([sol[0].rhs(), sol[1].rhs()])
    E = (t * A).exp().apply_map(lambda e: e.rectform().simplify_full())   # real form: cos, sin, cosh, sinh
    same = bool(all((a - b).simplify_full() == 0 for a, b in zip(flow, E * vector([x0, y0]))))
    group = bool(all((a - b).simplify_full() == 0 for a, b in zip(((s * A).exp() * (t * A).exp()).list(), ((s + t) * A).exp().list())))
    print(f"{name}: A = {A.list()}, exp(tA) = {[e.simplify_trig() for e in E.list()]}")
    print(f"   desolve flow equals exp(tA) (x0, y0): {same};  exp(sA) exp(tA) = exp((s+t)A): {group}")
