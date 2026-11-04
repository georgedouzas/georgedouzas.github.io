# Exercise exr-emw-dip-array: two parallel dipoles in phase, power factor cos^2(k d cos(phi)/2) in the equatorial plane.
ph = var('phi')
for name, kd in [('d = lambda/2', pi), ('d = lambda', 2 * pi)]:
    f = cos(kd * cos(ph) / 2)^2
    print(name, ': factor', f, '; values at phi = 0, pi/3, pi/2:', [f.subs(phi=v).simplify_full() for v in [0, pi / 3, pi / 2]])
