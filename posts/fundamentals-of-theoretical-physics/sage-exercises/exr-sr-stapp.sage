# Exercise exr-sr-stapp: Stapp's formula for sin(Omega) from the half-angle formula, symbolically in the half-rapidities.
var('x1 x2 th', domain='positive')
c1, s1, c2, s2 = cosh(x1/2), sinh(x1/2), cosh(x2/2), sinh(x2/2)
alpha = c1*c2 + s1*s2*cos(th)
C2 = alpha^2 + (s1*s2*sin(th))^2                 # cosh^2(zeta/2) = |coefficients|^2 of the rotation part
sinOm = 2*(s1*s2*sin(th))*alpha/C2
g1, g2 = cosh(x1), cosh(x2); g = cosh(x1)*cosh(x2) + sinh(x1)*sinh(x2)*cos(th)
v1, v2 = tanh(x1), tanh(x2)
stapp = g1*g2*v1*v2*(1 + g1 + g2 + g)*sin(th)/((1 + g1)*(1 + g2)*(1 + g))
for vals in [(0.3, 1.1, 0.7), (2.0, 0.5, 2.5), (1.0, 1.0, pi/2)]:
    d = dict(zip((x1, x2, th), vals))
    print("x1, x2, theta =", vals, "  sin(Omega) = %.12f  Stapp = %.12f" % (n(sinOm.subs(d)), n(stapp.subs(d))))
print("C^2 - (1 + gamma)/2 simplifies to:", (C2 - (1 + g)/2).simplify_full().trig_reduce().simplify_full())
