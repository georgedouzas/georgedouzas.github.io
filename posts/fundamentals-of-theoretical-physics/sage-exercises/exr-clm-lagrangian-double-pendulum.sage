# Exercise exr-clm-lagrangian-double-pendulum: the Lagrangian of the planar double pendulum from Cartesian positions.
t, m1, m2, l1, l2, g = var("t m1 m2 l1 l2 g", domain="positive")
th1, th2 = function("theta1")(t), function("theta2")(t)
x1, y1 = l1 * sin(th1), -l1 * cos(th1)
x2, y2 = x1 + l2 * sin(th2), y1 - l2 * cos(th2)
T = m1 / 2 * (diff(x1, t)**2 + diff(y1, t)**2) + m2 / 2 * (diff(x2, t)**2 + diff(y2, t)**2)
V = m1 * g * y1 + m2 * g * y2
L = T - V
expected = (m1 + m2) / 2 * l1**2 * diff(th1, t)**2 + m2 / 2 * l2**2 * diff(th2, t)**2 + m2 * l1 * l2 * cos(th1 - th2) * diff(th1, t) * diff(th2, t) + (m1 + m2) * g * l1 * cos(th1) + m2 * g * l2 * cos(th2)
print("L agrees with the stated formula:", bool((L - expected).simplify_full() == 0))
w1, w2 = var("w1 w2")
Tw = T.subs({diff(th1, t): w1, diff(th2, t): w2})
G = matrix([[diff(Tw, a, b) for b in (w1, w2)] for a in (w1, w2)]).simplify_full()
print("kinetic metric (Hessian of T in the angular velocities):"); print(G.subs({th1: var("q1"), th2: var("q2")}).apply_map(lambda e: e.trig_reduce()))
