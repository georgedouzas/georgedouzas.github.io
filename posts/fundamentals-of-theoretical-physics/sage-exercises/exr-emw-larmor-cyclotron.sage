# Exercise exr-emw-larmor-cyclotron (Macchi et al., Problem 10.1): damping of cyclotron motion (SI units).
t, k = var('t k', domain='positive'); En = function('En')(t)
print('solution:', desolve(diff(En, t) == -k * En, En, ivar=t))
e, me, eps0, c = 1.602e-19, 9.109e-31, 8.854e-12, 2.998e8
for B in [1, 10]:
    print('B = %d T: tau = 3 pi eps0 m^3 c^3/(e^4 B^2) = %.3g s' % (B, 3 * pi.n() * eps0 * me^3 * c^3 / (e^4 * B^2)))
