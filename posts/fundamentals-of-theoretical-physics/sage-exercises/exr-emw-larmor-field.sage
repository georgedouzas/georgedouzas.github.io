# Exercise exr-emw-larmor-field: Poynting flux of the radiation field of a slow charge.
th, ph, q, a, R = var('theta phi q a R', domain='positive')
n = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)]); acc = vector([0, 0, a])
E = q / (4 * pi * R) * n.cross_product(n.cross_product(acc)); B = n.cross_product(E)
print('E . n =', E.dot_product(n).simplify_full(), '; |B|^2 - |E|^2 =', (B.dot_product(B) - E.dot_product(E)).simplify_full())
print('R^2 S . n =', (R^2 * E.cross_product(B).dot_product(n)).simplify_full())
