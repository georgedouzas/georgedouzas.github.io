# Exercise exr-cfb-stokes (Maggiore, Sec. 9.5): a monochromatic plane wave along z with complex amplitude (E_x, E_y);
# Stokes parameters I, Q, U, V, the identity I^2 = Q^2 + U^2 + V^2, and the parameters of linear, circular and elliptic
# polarisation; the decomposition into the two circular polarisations.
a, b, d = var('a b d'); assume(a > 0, b > 0, d, 'real')
def stokes(Ex, Ey):
    S0 = abs(Ex)^2 + abs(Ey)^2; S1 = abs(Ex)^2 - abs(Ey)^2
    S2 = 2 * (Ex.conjugate() * Ey).real(); S3 = 2 * (Ex.conjugate() * Ey).imag()
    return [x.simplify_full() for x in (S0, S1, S2, S3)]
S0, S1, S2, S3 = stokes(a, b * exp(I * d))
print("general (a, b e^{i d}): I, Q, U, V =", S0, S1, S2, S3)
print("I^2 = Q^2 + U^2 + V^2:", bool((S0^2 - S1^2 - S2^2 - S3^2).simplify_full() == 0))
for name, (Ex, Ey) in [("linear along x", (1, 0)), ("linear at 45 deg", (1 / sqrt(2), 1 / sqrt(2))),
                       ("circular, positive helicity", (1 / sqrt(2), I / sqrt(2))), ("circular, negative helicity", (1 / sqrt(2), -I / sqrt(2))),
                       ("elliptic (1, 2i)/sqrt(5)", (1 / sqrt(5), 2 * I / sqrt(5)))]:
    print("%-28s I, Q, U, V =" % name, stokes(SR(Ex), SR(Ey)))
ep = vector([1, I]) / sqrt(2); em = vector([1, -I]) / sqrt(2)
Ev = vector([a, b * exp(I * d)])
cp = (ep.conjugate() * Ev).simplify_full(); cm = (em.conjugate() * Ev).simplify_full()
print("E = c+ e+ + c- e-:", bool(((cp * ep + cm * em) - Ev).simplify_full() == 0), "  |c+|^2 - |c-|^2 = V:", bool((cp * cp.conjugate() - cm * cm.conjugate() - S3).expand().simplify_full().simplify_trig() == 0))
