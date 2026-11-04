# Exercise exr-emm-gi-schrodinger: the Schroedinger operators are gauge covariant (one space dimension).
var('t x q')
psi = function('psi')(t, x); A = function('A')(t, x); phi = function('phi')(t, x); lam = function('lam')(t, x)
g = exp(I * q * lam)
kin = lambda f, AA: -I * diff(f, x) - q * AA * f
tim = lambda f, ph: I * diff(f, t) - q * ph * f
print('momentum operator:', (kin(g * psi, A + diff(lam, x)) - g * kin(psi, A)).simplify_full())
print('energy operator:', (tim(g * psi, phi - diff(lam, t)) - g * tim(psi, phi)).simplify_full())
