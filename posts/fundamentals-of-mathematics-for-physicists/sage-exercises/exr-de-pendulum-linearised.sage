# Exercise exr-de-pendulum-linearised: Jacobians of the damped pendulum at (0, 0) and (pi, 0), their classification, the
# matrix exponential of the linearisation at (0, 0), and the comparison with the numerical solution from (0.1, 0).
th, om, t = var("th om t")
gam = QQ(3) / 10
F = vector([om, -sin(th) - gam * om])
Jac = jacobian(F, [th, om])
for pt in [(0, 0), (pi, 0)]:
    A = Jac.subs(th=pt[0], om=pt[1])
    print("Jacobian at", pt, ":", A.list(), "| trace", A.trace(), "det", A.det(), "| eigenvalues", [numerical_approx(e) for e in A.change_ring(SR).eigenvalues()])
A0 = Jac.subs(th=0, om=0).change_ring(SR)
alpha, beta = -gam / 2, sqrt(1 - gam ** 2 / 4)                     # eigenvalues alpha +- i beta
E = exp(alpha * t) * (cos(beta * t) * identity_matrix(SR, 2) + sin(beta * t) / beta * (A0 - alpha * identity_matrix(SR, 2)))
print("exp(tA) at (0, 0) (closed form e^(alpha t)[cos(beta t) + sin(beta t)/beta (A - alpha)]):")
print(E)
print("d/dt exp(tA) = A exp(tA) and exp(0) = 1 (which identifies the exponential):", bool((E.derivative(t) - A0 * E).apply_map(lambda e: e.simplify_full()) == 0), bool(E.subs(t=0) == identity_matrix(2)))
raw = (t * A0).exp().apply_map(lambda e: e.rectform().simplify_full())
print("Sage's exp(tA) agrees with the closed form numerically at t = 1:", bool(abs(numerical_approx(raw.subs(t=1)[0, 0]) - numerical_approx(E.subs(t=1)[0, 0])) < 1e-12))
lin = E * vector([QQ(1) / 10, 0])
times = srange(0, 10.01, 0.5)
num = desolve_odeint([om, -sin(th) - gam * om], [0.1, 0], times, [th, om])
err = max(abs(float(lin[0].subs(t=tt)) - num[i][0]) for i, tt in enumerate(times))
print("largest difference in theta between the linear and the full solution over [0, 10]:", "%.2e" % err)
print("below 1e-3 (the cubic error of sin theta for theta = 0.1):", err < 1e-3)
