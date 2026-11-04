# Exercise exr-clm-co-spherical-sage: the spherical pendulum near the bottom in the coordinates u = alpha cos(phi),
# v = alpha sin(phi); the Lagrangian to second order is the isotropic oscillator.
t, s = var("t s"); m, g, l = var("m g l", domain="positive")
u, v = function("u")(t), function("v")(t)
# scale the displacement: u, v -> s u, s v (s a bookkeeping parameter), then expand to second order in s
al = sqrt((s * u)**2 + (s * v)**2); ph = arctan2(s * v, s * u)
L = m * l**2 / 2 * (diff(al, t)**2 + sin(al)**2 * diff(ph, t)**2) + m * g * l * cos(al)
assume(s > 0)
L2 = L.simplify_full().taylor(s, 0, 2).subs(s=1).simplify_full()
target = m * l**2 / 2 * (diff(u, t)**2 + diff(v, t)**2) - m * g * l / 2 * (u**2 + v**2) + m * g * l
print("L to second order:", L2)
print("equals m l^2 (u'^2 + v'^2)/2 - m g l (u^2 + v^2)/2 + m g l:", bool((L2 - target).simplify_full() == 0))
