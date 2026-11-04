# Exercise exr-emm-oblique-pressure (Macchi et al., Problem 8.11): radiation pressure on a mirror at oblique incidence.
I, c, th = var('I c theta', domain='positive')
# momentum per unit time per unit beam area along the normal: (I/c) cos(theta); reflection doubles it; a unit of beam area covers 1/cos(theta) of mirror
p = 2 * (I / c) * cos(th) * cos(th)
print('pressure =', p, '; at normal incidence:', p.subs({th: 0}), '; tangential momentum transfer: 0')
print('numbers: sunlight on a mirror at 60 degrees: %.2e Pa' % p.subs({I: 1361, c: 2.998e8, th: pi / 3}).n())
