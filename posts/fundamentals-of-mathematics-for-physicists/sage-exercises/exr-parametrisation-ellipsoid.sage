# Exercise exr-parametrisation-ellipsoid: (theta, phi) -> (A sin theta cos phi, B sin theta sin phi, C cos theta)
# parametrises the ellipsoid X^2/A^2 + Y^2/B^2 + Z^2/C^2 = 1; it is the spherical parametrisation followed by the
# invertible scaling diag(A, B, C), hence injective on (0, pi) x (0, 2 pi) with the same missing set as for the sphere.
th, ph = var("th ph", domain="real"); A, B, C = var("A B C", domain="real")
E = vector([A*sin(th)*cos(ph), B*sin(th)*sin(ph), C*cos(th)])
print("lies on the ellipsoid X^2/A^2 + Y^2/B^2 + Z^2/C^2 = 1:", bool((E[0]**2/A**2 + E[1]**2/B**2 + E[2]**2/C**2).simplify_full() == 1))
S = vector([sin(th)*cos(ph), sin(th)*sin(ph), cos(th)])
print("E = diag(A, B, C) S with S the spherical parametrisation:", bool(diagonal_matrix(SR, [A, B, C])*S == E))
print("det diag(A, B, C) =", diagonal_matrix(SR, [A, B, C]).det(), " -> invertible when A B C != 0, so E is injective exactly where S is")
J = jacobian(E, (th, ph))
print("d_theta E =", J.column(0)); print("d_phi E =", J.column(1))
print("|d_phi E|^2 =", (J.column(1)*J.column(1)).simplify_full().factor(), " vanishes exactly at the poles sin theta = 0")
print("the Jacobian columns are orthogonal only if A = B:", (J.column(0)*J.column(1)).simplify_full().factor())
