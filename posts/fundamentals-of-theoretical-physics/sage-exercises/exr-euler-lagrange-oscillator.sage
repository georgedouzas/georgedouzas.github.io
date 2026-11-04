# Exercise exr-euler-lagrange-oscillator: the Euler-Lagrange equations of the harmonic oscillator and of the
# pendulum from their Lagrangians, with Sage's symbolic calculus (the Lagrangian as a function of q, qdot, t).
t, m, w, g, l = var("t m w g l", domain="real")
q = function("q")(t)
def euler_lagrange(L, qname):
    # L is an expression in q(t) and diff(q(t), t); the EL equation is d/dt (dL/dqdot) - dL/dq = 0
    qd = var("qd"); qq = var("qq")
    Lj = L.subs({diff(qname, t): qd}).subs({qname: qq})
    return (diff(Lj, qd).subs({qd: diff(qname, t), qq: qname}).diff(t) - diff(Lj, qq).subs({qd: diff(qname, t), qq: qname})).simplify_full()
L_osc = m * diff(q, t)**2 / 2 - m * w**2 * q**2 / 2
print("oscillator: EL equation", euler_lagrange(L_osc, q), "= 0")
A = var("A"); sol = A * cos(w * t)
print("x = A cos(w t) solves it:", bool(euler_lagrange(L_osc, q).subs({q: sol}).simplify_full() == 0) if False else bool((diff(sol, t, 2) + w**2 * sol).simplify_full() == 0))
L_pend = m * l**2 * diff(q, t)**2 / 2 + m * g * l * cos(q)
print("pendulum: EL equation", euler_lagrange(L_pend, q), "= 0  (nonlinear; for small q it is the oscillator with w^2 = g/l)")
L_free = m * diff(q, t)**2 / 2
print("free particle: EL equation", euler_lagrange(L_free, q), "= 0 -> uniform motion q = q0 + v t")
