# Exercise exr-clb-anharmonic-period (Foulaadvand, Problem 3.2): period of oscillation in V = k |x|^p / p (m = k = 1) as a
# function of the amplitude A, from T = 4 int_0^A dx / sqrt(2 (V(A) - V(x))), compared with a numerical solution of
# the equation of motion, and the scaling T proportional to A^(1 - p/2).
def T_int(p, A):
    f = lambda u: 1 / sqrt(2 * (A^p - (A * u)^p) / p) * A    # x = A u
    return 4 * numerical_integral(f, 0, 1)[0]
def T_ode(p, A):
    t = var('t'); xs = var('xs'); vs = var('vs')
    sol = desolve_odeint([vs, -sgn(xs) * abs(xs)^(p - 1)], [A, 0], srange(0, 40, 0.001), [xs, vs])
    # the velocity first changes sign from negative to positive at x = -A, after half a period
    ts = srange(0, 40, 0.001)
    for i in range(1, len(ts)):
        if sol[i - 1][1] < 0 <= sol[i][1]:
            return 2 * ts[i]
for p in (1, 2, 3, 4):
    vals = [(A, T_int(p, A)) for A in (0.5, 1.0, 2.0)]
    print("p = %d: T(A) for A = 0.5, 1, 2: %s; T(2)/T(1) = %.4f, 2^(1 - p/2) = %.4f; ODE at A = 1: %.3f" %
          (p, [round(v, 4) for _, v in vals], vals[2][1] / vals[1][1], 2^(1 - p / 2), T_ode(p, 1.0)))
print("p = 1 closed form 4 sqrt(2 A) at A = 1:", (4 * sqrt(2.0)).n(digits=6))
