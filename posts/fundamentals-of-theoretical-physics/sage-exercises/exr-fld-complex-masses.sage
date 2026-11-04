# Exercise exr-fld-complex-masses: two real scalars u, v with masses m1, m2 combined into phi = (u + i v)/sqrt(2).
# The Lagrangian (1/2)(du^2 - m1^2 u^2) + (1/2)(dv^2 - m2^2 v^2) in 1+1 dimensions, signature (+, -), changes under the
# rotation (u, v) -> (u cos a - v sin a, u sin a + v cos a) by a multiple of (m1^2 - m2^2): the U(1) needs m1 = m2.
t, x, a, m1, m2 = var("t x a m1 m2", domain="real")
u, v = function("u")(t, x), function("v")(t, x)
def L(u, v):
    return (diff(u, t)**2 - diff(u, x)**2 - m1**2 * u**2) / 2 + (diff(v, t)**2 - diff(v, x)**2 - m2**2 * v**2) / 2
ur, vr = u * cos(a) - v * sin(a), u * sin(a) + v * cos(a)
dL = (L(ur, vr) - L(u, v)).simplify_full()
print("change of L under the rotation:", dL.factor())
print("it vanishes for m1 = m2:", bool(dL.subs(m2=m1).simplify_full() == 0))
phi, phib = (u + I * v) / sqrt(2), (u - I * v) / sqrt(2)
m = var("m")
Lc = diff(phib, t) * diff(phi, t) - diff(phib, x) * diff(phi, x) - m**2 * phib * phi
print("for m1 = m2 = m: L = d phibar d phi - m^2 |phi|^2:", bool((Lc - L(u, v).subs(m1=m, m2=m)).simplify_full() == 0))
