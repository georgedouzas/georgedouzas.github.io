# Exercise exr-cm-s3-volume: the volume of SU(2) = S^3 computed in two charts. Euler angles: sqrt(det G) = sin(theta)/8
# over (0, pi) x (0, 2 pi) x (0, 4 pi). Stereographic chart of S^3: Gram matrix 4/(1 + rho^2)^2 times the identity,
# sqrt(det G) = 8/(1 + rho^2)^3 over R^3. Both give 2 pi^2.
th, ph, ps = var("th ph ps", domain="real")
G_euler = matrix(SR, [[1, 0, 0], [0, 1, cos(th)], [0, cos(th), 1]])/4
dens = sin(th)/8
print("sqrt(det G_euler) = sin(theta)/8 on 0 < theta < pi:", bool((dens**2 - G_euler.det()).simplify_full() == 0))
vol_euler = integrate(integrate(integrate(dens, th, 0, pi), ph, 0, 2*pi), ps, 0, 4*pi)
print("volume from the Euler angles:", vol_euler)
u, v, w = var("u v w", domain="real")
rho2 = u**2 + v**2 + w**2
E = vector([2*u/(1 + rho2), 2*v/(1 + rho2), 2*w/(1 + rho2), (rho2 - 1)/(1 + rho2)])
J = jacobian(E, (u, v, w))
G = (J.transpose()*J).simplify_full()
print("Gram matrix of the stereographic chart of S^3 is 4/(1 + rho^2)^2 times the identity:", bool((G - 4/(1 + rho2)**2*identity_matrix(SR, 3)).simplify_full() == 0))
print("det G =", G.det().simplify_full().factor())
rho = var("rho", domain="real"); assume(rho > 0)
dens3 = 8/(1 + rho**2)**3                                  # sqrt(det G) as a function of rho = sqrt(u^2 + v^2 + w^2)
print("sqrt(det G) = 8/(1 + rho^2)^3:", bool((dens3**2 - G.det().subs(u=rho, v=0, w=0)).simplify_full() == 0))
vol_ste = integrate(4*pi*rho**2*dens3, rho, 0, oo)         # shells of radius rho in R^3
print("volume from the stereographic chart:", vol_ste)
print("the two charts agree, vol(S^3) = 2 pi^2:", bool(vol_euler == vol_ste) and bool(vol_euler == 2*pi**2))
