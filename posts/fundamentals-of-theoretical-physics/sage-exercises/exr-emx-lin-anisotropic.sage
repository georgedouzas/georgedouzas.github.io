# Exercise exr-emx-lin-anisotropic: angle between D and E in a uniaxial medium.
th = var('th')
e1, e2 = 2.75, 2.21
thD = arctan(e1 / e2 * tan(th))
diffang = thD - th
th0 = arctan(sqrt(e2 / e1))
print('maximum at tan(theta) = sqrt(e2/e1): theta =', n(th0 * 180 / pi, digits=4), 'deg')
print('derivative there:', n(diff(diffang, th).subs({th: th0}), digits=3))
print('largest angle between D and E:', n(diffang.subs({th: th0}) * 180 / pi, digits=4), 'deg')
