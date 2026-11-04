# Exercise exr-ssg-wz-domain-wall: for real W = m phi^2/2 + y phi^3/6 (m = y = 1) and a static field phi(x) along one
# direction, the energy per area E = int (phi'^2 + W'^2) dx = int (phi' - W')^2 dx + 2 int phi' W' dx is bounded below by
# 2|W(phi(+oo)) - W(phi(-oo))| (Bogomolnyi). The kink phi(x) = -2/(1 + exp(-x)) interpolating between the vacua 0 and -2
# solves phi' = W'(phi) and saturates the bound E = 2|W(-2) - W(0)| = 4/3.
x, p = var("x p"); W = p^2 / 2 + p^3 / 6; Wp = diff(W, p)
phi = -2 / (1 + exp(-x))
print("[PASS] the kink solves the first-order equation phi' = W'(phi):", bool((diff(phi, x) - Wp.subs(p=phi)).simplify_full() == 0))
print("[PASS] it connects the two supersymmetric vacua: phi(-oo) = 0, phi(+oo) = -2:", bool(limit(phi, x=-oo) == 0) and bool(limit(phi, x=oo) == -2))
E = integrate((diff(phi, x)^2 + Wp.subs(p=phi)^2).simplify_full(), x, -oo, oo)
bound = 2 * abs(W.subs(p=-2) - W.subs(p=0))
print("energy per area E =", E, "; Bogomolnyi bound 2|Delta W| =", bound)
print("[PASS] the kink saturates the bound E = 2|W(-2) - W(0)| = 4/3:", bool(E == bound) and bool(bound == 4 / 3))
