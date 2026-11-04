# Exercise exr-emx-mag-sphere-check: field of a uniformly magnetised sphere from its vector potential.
x, y, z, M, a = var('x y z M a')
assume(a > 0)
X = vector([x, y, z]); ez = vector([0, 0, 1])
r = sqrt(x^2 + y^2 + z^2)
def curl(A):
    return vector([diff(A[2], y) - diff(A[1], z), diff(A[0], z) - diff(A[2], x), diff(A[1], x) - diff(A[0], y)])
Ain = M / 3 * ez.cross_product(X)
Aout = M * a^3 / 3 * ez.cross_product(X) / r^3
Bin = curl(Ain); Bout = curl(Aout).simplify_full()
print('B inside:', Bin)
m = 4 * pi * a^3 / 3 * M
dip = vector([3 * x * z, 3 * y * z, 3 * z^2 - r^2]) * m / (4 * pi * r^5)
print('B outside equals the dipole field of m = (4 pi a^3/3) M:', bool((Bout - dip).simplify_full() == 0))
eq_in = Bin.subs(x=a, y=0, z=0); eq_out = Bout.subs(x=a, y=0, z=0)
print('B_z just inside / outside at the equator:', eq_in[2], ',', eq_out[2].simplify_full())
print('jump B_z(in) - B_z(out) =', (eq_in[2] - eq_out[2]).simplify_full(), '= |K_b| = M sin(pi/2)')
