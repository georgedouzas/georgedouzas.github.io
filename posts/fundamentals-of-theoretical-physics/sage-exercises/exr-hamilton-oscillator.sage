# Exercise exr-hamilton-oscillator: Legendre transform of the oscillator Lagrangian, Hamilton's equations, and the
# conservation of the Hamiltonian along their solutions, with Sage.
t, m, w = var("t m w", domain="real"); assume(m > 0); assume(w > 0)
q, qd, p = var("q qd p", domain="real")
L = m * qd**2 / 2 - m * w**2 * q**2 / 2
p_of_qd = diff(L, qd); print("canonical momentum p = dL/dqdot =", p_of_qd)
qd_of_p = solve(p_of_qd == p, qd)[0].rhs()
H = (p * qd - L).subs(qd=qd_of_p).simplify_full(); print("Hamiltonian H = p qdot - L =", H)
print("Hamilton's equations: qdot =", diff(H, p), ", pdot =", -diff(H, q))
Q = function("Q")(t); P = function("P")(t)
sol = desolve_system([diff(Q, t) == diff(H, p).subs({p: P}), diff(P, t) == -diff(H, q).subs({q: Q})], [Q, P], ics=[0, var("A"), 0], ivar=t)
print("solution with Q(0) = A, P(0) = 0:", [s.rhs().simplify_full() for s in sol])
Hs = H.subs({q: sol[0].rhs(), p: sol[1].rhs()}).simplify_full()
print("H along the solution:", Hs, " (constant)")
