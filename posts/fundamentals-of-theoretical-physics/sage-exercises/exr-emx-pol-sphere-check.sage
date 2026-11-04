# Exercise exr-emx-pol-sphere-check: potential of a uniformly polarised sphere in spherical coordinates.
r, th, P, a = var('r theta P a', domain='positive')
def lap(f):
    return (diff(r^2 * diff(f, r), r) / r^2 + diff(sin(th) * diff(f, th), th) / (r^2 * sin(th))).simplify_full()
phi_in = P / 3 * r * cos(th)
phi_out = P * a^3 * cos(th) / (3 * r^2)
print('Laplacian inside :', lap(phi_in))
print('Laplacian outside:', lap(phi_out))
print('continuity at r = a:', bool((phi_in - phi_out).subs(r=a) == 0))
jump = (-diff(phi_out, r) + diff(phi_in, r)).subs(r=a).simplify_full()
print('jump of -d_r phi:', jump, '; equals P cos(theta):', bool(jump == P * cos(th)))
print('E inside, z component: -d_z(P z/3) =', -P / 3)
