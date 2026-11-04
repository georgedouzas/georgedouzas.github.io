# Exercise exr-emx-mm-magnetic-sphere: magnetisable sphere in a uniform field, scalar potential psi with H = -grad psi.
r, th, B0, a, mu = var('r theta B0 a mu', domain='positive')
pin = -3 * B0 / (mu + 2) * r * cos(th)
pout = -B0 * r * cos(th) + (mu - 1) / (mu + 2) * B0 * a^3 * cos(th) / r^2
Hth = lambda p: -diff(p, th) / r
print('H_theta continuous:', bool((Hth(pin) - Hth(pout)).subs(r=a).simplify_full() == 0))
print('B_r continuous:', bool((mu * diff(pin, r) - diff(pout, r)).subs(r=a).simplify_full() == 0))
print('B inside along z:', (mu * 3 * B0 / (mu + 2)), '= 3 mu/(mu + 2) B0')
