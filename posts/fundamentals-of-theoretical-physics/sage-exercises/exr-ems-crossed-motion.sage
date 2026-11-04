# Exercise exr-ems-crossed-motion (Macchi et al., Problem 5.6): charge from rest in E = E e_y, B = B e_z.
t, q, m, E, B = var('t q m E B', domain='positive')
vx, vy = function('vx')(t), function('vy')(t)
w = q * B / m
# m vx' = q vy B,  m vy' = q E - q vx B, from rest
sol = desolve_system([diff(vx, t) == w * vy, diff(vy, t) == q * E / m - w * vx], [vx, vy], ics=[0, 0, 0], ivar=t)
u = var('u')
xs = integrate(sol[0].rhs().subs(t=u), u, 0, t); ys = integrate(sol[1].rhs().subs(t=u), u, 0, t)
print("x(t) =", xs.simplify_full()); print("y(t) =", ys.simplify_full())
print("matches the cycloid:", bool((xs - E / B * (t - sin(w * t) / w)).simplify_full() == 0), bool((ys - E / (B * w) * (1 - cos(w * t))).simplify_full() == 0))
print("largest y:", (2 * E / (B * w)).simplify_full())
h, V = var('h V', domain='positive')
print("electrons stay below h for B >", solve(2 * m * (V / h) / (q * B^2) == h, B))
