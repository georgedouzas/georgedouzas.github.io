# Exercise exr-ems-legendre-outside: potential outside the sphere with boundary values V0 cos(theta)^2, and the charge.
r, th, V0, R = var('r theta V0 R'); assume(R > 0, r > 0)
P2 = (3 * cos(th)^2 - 1) / 2
phi_out = V0 / 3 * R / r + 2 * V0 / 3 * R^3 / r^3 * P2
phi_in = V0 / 3 + 2 * V0 / 3 * r^2 / R^2 * P2
lap = (diff(r^2 * diff(phi_out, r), r) / r^2 + diff(sin(th) * diff(phi_out, th), th) / (r^2 * sin(th))).simplify_full()
print("Laplacian outside:", lap)
print("boundary values match:", bool((phi_out.subs(r=R) - V0 * cos(th)^2).simplify_full() == 0), bool((phi_in.subs(r=R) - V0 * cos(th)^2).simplify_full() == 0))
sigma = (-diff(phi_out, r) + diff(phi_in, r)).subs(r=R).simplify_full()
print("surface charge sigma(theta) =", sigma)
print("total charge:", integrate(2 * pi * R^2 * sigma * sin(th), th, 0, pi).simplify_full())
