# Exercise exr-velocity-stereographic: the coordinate velocities d_u E and d_v E of the inverse stereographic chart:
# both are tangent to the sphere, orthogonal to each other, and of the same length 2/(1 + u^2 + v^2), which is 2 at the
# south pole and tends to 0 towards the north pole.
u, v = var("u v", domain="real")
rho2 = u**2 + v**2
E = vector([2*u/(1 + rho2), 2*v/(1 + rho2), (rho2 - 1)/(1 + rho2)])
J = jacobian(E, (u, v))
Eu, Ev = J.column(0), J.column(1)
print("d_u E =", Eu.apply_map(lambda e: e.simplify_full().factor()))
print("d_v E =", Ev.apply_map(lambda e: e.simplify_full().factor()))
print("both tangent to the sphere, E . d_u E = E . d_v E = 0:", bool((E*Eu).simplify_full() == 0) and bool((E*Ev).simplify_full() == 0))
print("orthogonal to each other, d_u E . d_v E = 0:", bool((Eu*Ev).simplify_full() == 0))
print("|d_u E|^2 = |d_v E|^2 = 4/(1 + u^2 + v^2)^2:", bool(((Eu*Eu) - 4/(1 + rho2)**2).simplify_full() == 0) and bool(((Ev*Ev) - 4/(1 + rho2)**2).simplify_full() == 0))
print("length at the origin (south pole):", sqrt((Eu*Eu).subs(u=0, v=0)), "; limit of the length as u -> oo along v = 0:", limit(sqrt((Eu*Eu).simplify_full()).subs(v=0), u=oo))
print("the Jacobian has rank 2 everywhere: the stereographic parametrisation is regular on the whole plane")
