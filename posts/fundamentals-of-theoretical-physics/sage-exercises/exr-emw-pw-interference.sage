# Exercise exr-emw-pw-interference: two plane waves polarised along e_z crossing at angles +-alpha.
t, x, y, k, al, E0 = var('t x y k alpha E0'); assume(k > 0)
n1 = vector([cos(al), sin(al), 0]); n2 = vector([cos(al), -sin(al), 0]); ez = vector([0, 0, 1])
p1 = k * (x * cos(al) + y * sin(al)) - k * t; p2 = k * (x * cos(al) - y * sin(al)) - k * t
E = E0 * (cos(p1) + cos(p2)) * ez
B = E0 * cos(p1) * n1.cross_product(ez) + E0 * cos(p2) * n2.cross_product(ez)
u = (E.dot_product(E) + B.dot_product(B)) / 2
avg = (k / (2 * pi) * integrate(u.expand(), t, 0, 2 * pi / k)).simplify_full()
print('time-averaged energy density:', avg)
print('equals E0^2 (1 + cos(alpha)^2 cos(2 k y sin(alpha))):', bool((avg - E0^2 * (1 + cos(al)^2 * cos(2 * k * y * sin(al)))).simplify_full().simplify_trig() == 0))
