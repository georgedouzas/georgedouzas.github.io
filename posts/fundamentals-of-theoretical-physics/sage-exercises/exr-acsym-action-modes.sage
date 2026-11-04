# Exercise exr-acsym-action-modes: a real scalar on the interval [0, l] with phi(0) = phi(l) = 0, expanded in the modes
# phi = sum_n q_n(t) sin(n pi x / l). The spatial integral of the Lagrangian density
# (1/2) phi_t^2 - (1/2) phi_x^2 - (1/2) m^2 phi^2 becomes a sum of harmonic-oscillator Lagrangians. Three modes.
x, l, m = var('x l m'); assume(l > 0)
N = 3
q = [var('q%d' % n) for n in range(1, N + 1)]; qd = [var('qd%d' % n) for n in range(1, N + 1)]   # q_n and dq_n/dt
s = [sin(n * pi * x / l) for n in range(1, N + 1)]
phi = sum(q[i] * s[i] for i in range(N)); phit = sum(qd[i] * s[i] for i in range(N))
Ldens = phit^2 / 2 - diff(phi, x)^2 / 2 - m^2 * phi^2 / 2
Lagr = integrate(Ldens.expand(), x, 0, l).simplify_full()
w2 = [(n * pi / l)^2 + m^2 for n in range(1, N + 1)]
target = sum((l / 2) * (qd[i]^2 / 2 - w2[i] * q[i]^2 / 2) for i in range(N))
print("L = int_0^l Lagrangian density dx =", Lagr.expand())
print("L = sum_n (l/2) (qd_n^2/2 - omega_n^2 q_n^2/2) with omega_n^2 = (n pi/l)^2 + m^2:", bool((Lagr - target).simplify_full() == 0))
print("no cross terms q_i q_j or qd_i qd_j for i != j (the modes are orthogonal):", all(bool(diff(Lagr, q[i], q[j]) == 0) and bool(diff(Lagr, qd[i], qd[j]) == 0) for i in range(N) for j in range(N) if i != j))
for i in range(N):
    print("  Euler-Lagrange equation of q_%d:  (l/2) (qdd_%d + (%s) q_%d) = 0" % (i + 1, i + 1, w2[i], i + 1))
