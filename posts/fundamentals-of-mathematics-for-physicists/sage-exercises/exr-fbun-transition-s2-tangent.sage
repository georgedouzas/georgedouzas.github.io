# Exercise exr-fbun-transition-s2-tangent: the transition function of TS^2 between the two stereographic charts
# z (from the north pole) and w = 1/z (from the south pole) is the Jacobian of the change of chart, multiplication by
# dw/dz = -1/z^2; on the equator |z| = 1 it is a rotation that winds -2 times as one goes round once.
x, y = var("x y", domain="real")
r2 = x ** 2 + y ** 2
u, v = x / r2, -y / r2                                       # w = 1/z = conj(z)/|z|^2
J = matrix([[diff(u, x), diff(u, y)], [diff(v, x), diff(v, y)]]).simplify_full()
print("Jacobian of z -> 1/z:"); print(J)
print("determinant:", J.det().simplify_full(), "(positive: the charts have the same orientation)")
dwdz = -1 / (x + I * y) ** 2
print("J is multiplication by dw/dz = -1/z^2 (Cauchy-Riemann form):",
      bool((J[0, 0] - dwdz.real()).simplify_full() == 0 and (J[1, 0] - dwdz.imag()).simplify_full() == 0
           and (J[0, 1] + J[1, 0]).simplify_full() == 0 and (J[0, 0] - J[1, 1]).simplify_full() == 0))
t = var("t", domain="real")
g = (-1 / exp(I * t) ** 2)                                   # dw/dz on the equator z = e^{it}
wind = integrate((diff(g, t) / g).simplify_full(), t, 0, 2 * pi) / (2 * pi * I)
print("winding number of the transition function around the equator:", wind)
