# Exercise exr-emm-gi-noether-check: a gauge-invariant conserved quantity on a circular orbit in B e_z.
var('t'); m, q, B, R, X0, Y0 = var('m q B R X0 Y0', domain='positive')
w = q * B / m
x = X0 + R * cos(w * t); y = Y0 - R * sin(w * t)          # clockwise for q B > 0
print('orbit solves m x\'\' = q B y\', m y\'\' = -q B x\':', bool((m * diff(x, t, 2) - q * B * diff(y, t)).simplify_full() == 0), bool((m * diff(y, t, 2) + q * B * diff(x, t)).simplify_full() == 0))
py_sym = m * diff(y, t) + q * B / 2 * x; py_landau = m * diff(y, t) + q * B * x
print('symmetric gauge: p_y =', py_sym.simplify_full(), '(oscillates); p_y + qBx/2 =', (py_sym + q * B / 2 * x).simplify_full())
print('Landau gauge: p_y =', py_landau.simplify_full(), '= q B X0')
