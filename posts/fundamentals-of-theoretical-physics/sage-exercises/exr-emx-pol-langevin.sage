# Exercise exr-emx-pol-langevin: the Langevin function and the size of pE/kT for water.
x = var('x')
L = coth(x) - 1 / x
print('series:', L.series(x, 6))
print('limit x -> oo:', limit(L, x=oo))
p = 6.2e-30        # dipole moment of water, C m
E = 1e6            # V/m
kB, T = 1.380649e-23, 300
print('pE/kT =', n(p * E / (kB * T), digits=3))
