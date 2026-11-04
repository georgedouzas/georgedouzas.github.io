# Exercise exr-ems-helmholtz (Macchi et al., Problem 5.14): two coaxial loops of radius a at z = +-b, same current I.
I, a, b, z = var('I a b z', domain='positive')
Bloop = lambda zz: I * a^2 / (2 * (a^2 + zz^2)^(3/2))
B = Bloop(z - b) + Bloop(z + b)
d = [diff(B, z, k).subs(z=0).simplify_full() for k in range(1, 5)]
print("first and third derivatives at 0:", d[0], d[2])
print("second derivative at 0:", d[1].factor())
print("b with vanishing second derivative:", solve(d[1] == 0, b))
print("B(0) at b = a/2:", B.subs(z=0, b=a / 2).simplify_full(), "= (4/5)^(3/2) I/a:", bool((B.subs(z=0, b=a / 2) - (4 / 5)^(3/2) * I / a).simplify_full() == 0))
print("fourth derivative at 0, b = a/2:", d[3].subs(b=a / 2).simplify_full())
