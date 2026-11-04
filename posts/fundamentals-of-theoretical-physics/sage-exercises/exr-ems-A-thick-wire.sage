# Exercise exr-ems-A-thick-wire: the vector potential A_z of a thick wire, inside and outside.
I, a, s = var('I a s', domain='positive')
Ain = -I * s^2 / (4 * pi * a^2); Aout = -I / (4 * pi) * (1 + 2 * log(s / a))
print("values at s = a:", Ain.subs(s=a), Aout.subs(s=a), "; derivatives:", diff(Ain, s).subs(s=a), diff(Aout, s).subs(s=a).simplify_full())
print("B_phi = -dA_z/ds inside:", -diff(Ain, s), "; outside:", -diff(Aout, s))
