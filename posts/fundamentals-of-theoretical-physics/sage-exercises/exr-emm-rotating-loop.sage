# Exercise exr-emm-rotating-loop (Macchi et al., Problem 6.12): small loop rotating at the centre of a large one.
t, I, a, b, R, w = var('t I a b R w', domain='positive')
Phi = pi * b^2 * I / (2 * a) * cos(w * t)
emf = -diff(Phi, t); Ib = emf / R
print('induced current:', Ib)
Pav = (w / (2 * pi) * integrate(Ib^2 * R, t, 0, 2 * pi / w)).simplify_full()
print('average Joule power:', Pav, '; average torque = P/w =', (Pav / w).simplify_full())
