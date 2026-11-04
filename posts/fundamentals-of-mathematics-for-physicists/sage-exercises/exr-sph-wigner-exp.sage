# Exercise exr-sph-wigner-exp: exp(-i theta J_y) for spin 1/2 and spin 1 from the series (J_y^2 = 1/4, resp. J_y^3 = J_y),
# compared with d^{1/2} and d^1.
th = var("theta")
sig2 = matrix(SR, [[0, -I], [I, 0]]); Jh = sig2 / 2
dh = (cos(th / 2) * identity_matrix(2) - 2 * I * sin(th / 2) * Jh).simplify_full()
print("spin 1/2: J_y^2 = 1/4:", bool(Jh ** 2 == identity_matrix(2) / 4), "; exp(-i theta J_y) =", dh.list())
print("  equals d^{1/2}(theta):", bool((dh - matrix(SR, [[cos(th / 2), -sin(th / 2)], [sin(th / 2), cos(th / 2)]])).simplify_full() == 0))
Jp = matrix(SR, [[0, sqrt(2), 0], [0, 0, sqrt(2)], [0, 0, 0]]); Jy = (Jp - Jp.transpose()) / (2 * I)
print("spin 1: J_y^3 = J_y:", bool((Jy ** 3 - Jy).simplify_full() == 0))
d1s = (identity_matrix(3) - I * Jy * sin(th) - Jy ** 2 * (1 - cos(th))).simplify_full()
d1 = matrix(SR, [[(1 + cos(th)) / 2, -sin(th) / sqrt(2), (1 - cos(th)) / 2], [sin(th) / sqrt(2), cos(th), -sin(th) / sqrt(2)], [(1 - cos(th)) / 2, sin(th) / sqrt(2), (1 + cos(th)) / 2]])
print("  1 - i J_y sin theta - J_y^2 (1 - cos theta) = d^1(theta):", bool((d1s - d1).simplify_full() == 0))
ode = (d1s.derivative(th) + I * Jy * d1s).simplify_full()
print("  it satisfies d' = -i J_y d:", ode.is_zero())
print("d^{1/2}(2 pi) = -1 while d^1(2 pi) = 1:", bool(dh.subs({th: 2 * pi}) == -identity_matrix(2)) and bool(d1.subs({th: 2 * pi}) == identity_matrix(3)))
