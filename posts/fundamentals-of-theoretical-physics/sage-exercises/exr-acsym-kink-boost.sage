# Exercise exr-acsym-kink-boost: the moving kink phi = f(gamma (x - u t)) of V = (lam/4)(phi^2 - v^2)^2 in 1+1
# dimensions, f(y) = v tanh(a y), a = v sqrt(lam/2). Its energy E = int T^{tt} dx and momentum P = int T^{tx} dx are
# gamma M and gamma M u with M the rest energy, so E^2 - P^2 = M^2: the kink moves like a relativistic particle.
t, x, y, p = var('t x y p', domain='real'); lam, v = var('lam v', domain='positive'); u = var('u', domain='real')
assume(u > -1, u < 1)
g = 1 / sqrt(1 - u^2); a = v * sqrt(lam / 2)
V = lam / 4 * (p^2 - v^2)^2
phi = v * tanh(a * g * (x - u * t))
print("phi_tt - phi_xx + V'(phi) =", (diff(phi, t, 2) - diff(phi, x, 2) + diff(V, p).subs(p=phi)).simplify_full())
Ttt = (diff(phi, t)^2 / 2 + diff(phi, x)^2 / 2 + V.subs(p=phi))           # T^{tt}
Ttx = -diff(phi, t) * diff(phi, x)                                       # T^{tx} = d^t phi d^x phi
M = 2 * sqrt(2) / 3 * sqrt(lam) * v^3                                     # rest energy of exr-acsym-solutions-kink
# at t = 0, substitute x = y/(a gamma): both densities become multiples of sech(y)^4
E = integrate((Ttt.subs(t=0).subs(x=y / (a * g)) / (a * g)).simplify_full(), y, -oo, oo)
P = integrate((Ttx.subs(t=0).subs(x=y / (a * g)) / (a * g)).simplify_full(), y, -oo, oo)
print("E =", E.simplify_full(), "  equals gamma M:", bool((E - g * M).simplify_full() == 0))
print("P =", P.simplify_full(), "  equals gamma M u:", bool((P - g * M * u).simplify_full() == 0))
print("E^2 - P^2 = M^2:", bool((E^2 - P^2 - M^2).simplify_full() == 0))
