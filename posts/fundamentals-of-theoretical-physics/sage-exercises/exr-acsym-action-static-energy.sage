# Exercise exr-acsym-action-static-energy: the energy E[phi] = int_0^1 (phi'^2 + m^2 phi^2)/2 dx of the varied static
# fields phi = sinh(m x)/sinh(m) + eps sin(pi x) of figure field_action, as an exact function of eps and m.
x, eps = var('x eps', domain='real'); m = var('m'); assume(m > 0)
phicl = sinh(m * x) / sinh(m)
phi = phicl + eps * sin(pi * x)
E = integrate((diff(phi, x)^2 + m^2 * phi^2) / 2, x, 0, 1)
E0 = E.subs(eps=0); E1 = diff(E, eps).subs(eps=0); E2 = diff(E, eps, 2) / 2
print("E(eps) is a polynomial of degree 2 in eps:", bool(diff(E, eps, 3).simplify_full() == 0))
print("E(0) =", E0.simplify_full(), "  equals (m/2) coth m:", bool((E0 - m * cosh(m) / (2 * sinh(m))).exponentialize().simplify_full() == 0))
print("linear coefficient E'(0) =", E1.simplify_full())
print("quadratic coefficient =", E2.simplify_full(), "  equals (pi^2 + m^2)/4:", bool((E2 - (pi^2 + m^2) / 4).simplify_full() == 0))
print("so E(eps) = (m/2) coth m + eps^2 (pi^2 + m^2)/4: a minimum at eps = 0, for every m > 0")
