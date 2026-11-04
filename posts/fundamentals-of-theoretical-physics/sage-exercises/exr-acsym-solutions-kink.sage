# Exercise exr-acsym-solutions-kink: the static kink of the double-well potential V = (lambda/4)(phi^2 - v^2)^2 in
# 1+1 dimensions. The static Euler-Lagrange equation is phi'' = V'(phi); the kink is phi = v tanh(a x) with
# a = v sqrt(lambda/2), and its energy is E = int (phi'^2/2 + V) dx = (2 sqrt 2/3) sqrt(lambda) v^3.
x, p = var('x p', domain='real'); lam, v = var('lam v'); assume(lam > 0, v > 0)
V = lam / 4 * (p^2 - v^2)^2
a = v * sqrt(lam / 2)
phi = v * tanh(a * x)
eq = (diff(phi, x, 2) - diff(V, p).subs(p=phi)).simplify_full()
print("phi'' - V'(phi) =", eq)
print("first-order (Bogomolny) equation phi' = sqrt(2 V(phi)) = (sqrt(lam/2)) (v^2 - phi^2):", bool((diff(phi, x) - sqrt(lam / 2) * (v^2 - phi^2)).simplify_full() == 0))
dens = (diff(phi, x)^2 / 2 + V.subs(p=phi)).simplify_full()
print("energy density = phi'^2 (gradient and potential energies are equal):", bool((dens - diff(phi, x)^2).simplify_full() == 0))
y = var('y')
E = integrate((a * v)^2 * sech(y)^4, y, -oo, oo) / a                          # substitute y = a x
print("E =", E.simplify_full(), "  equals (2 sqrt 2/3) sqrt(lam) v^3:", bool((E - 2 * sqrt(2) / 3 * sqrt(lam) * v^3).simplify_full() == 0))
print("the kink interpolates between the vacua: limits", limit(phi, x=-oo), "and", limit(phi, x=oo))
