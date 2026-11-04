# Exercise exr-cm-torus-curve-length: lengths on the torus R = 2, r = 1 from its Gram matrix G = diag(r^2, (R + r cos a)^2):
# a tube circle (2 pi r), the axis circles through a = 0 and a = pi (2 pi (R +- r)), and the closed curve (t, 2t),
# whose speed is sqrt(G_aa + 4 G_bb) and whose length is computed numerically and bounded.
R, r = 2, 1
a, b, t = var("a b t", domain="real")
E = vector([(R + r*cos(a))*cos(b), (R + r*cos(a))*sin(b), r*sin(a)])
J = jacobian(E, (a, b)); G = (J.transpose()*J).simplify_full()
print("Gram matrix:", G.list())
print("length of a tube circle, integral of sqrt(G_aa) over a in (0, 2 pi):", integrate(sqrt(G[0, 0]), a, 0, 2*pi))
print("axis circles at a = 0 (outer equator) and a = pi (inner equator):", integrate(sqrt(G[1, 1].subs(a=0)), b, 0, 2*pi), integrate(sqrt(G[1, 1].subs(a=pi)), b, 0, 2*pi))
# gamma(t) = (t, 2t): speed^2 = G_aa adot^2 + 2 G_ab adot bdot + G_bb bdot^2 with (adot, bdot) = (1, 2)
speed2 = (G[0, 0] + 4*G[0, 1] + 4*G[1, 1]).subs(a=t, b=2*t).simplify_full()
print("speed squared along (t, 2t):", speed2.factor())
Edot = diff(E.subs(a=t, b=2*t), t)
print("agrees with |dE/dt|^2 computed in R^3:", bool((Edot*Edot - speed2).simplify_full() == 0))
L = numerical_integral(sqrt(speed2), 0, 2*pi)
print("length of the closed curve (t, 2t):", L[0], " (error estimate", L[1], ")")
print("bounds 2 pi sqrt(1 + 4 (R - r)^2) <= L <= 2 pi sqrt(1 + 4 (R + r)^2):", N(2*pi*sqrt(1 + 4*(R - r)**2)), "<=", L[0], "<=", N(2*pi*sqrt(1 + 4*(R + r)**2)))
