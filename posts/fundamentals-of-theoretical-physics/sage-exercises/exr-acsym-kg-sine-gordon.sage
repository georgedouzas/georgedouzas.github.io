# Exercise exr-acsym-kg-sine-gordon: the sine-Gordon Lagrangian L = pt^2/2 - px^2/2 - (1 - cos p) in the jet variables
# (p, pt, px) of 1+1 dimensions, its Euler-Lagrange equation phi_tt - phi_xx + sin(phi) = 0, and the static kink
# phi = 4 arctan(exp(x)) and its boost 4 arctan(exp(gamma (x - u t))).
t, x = var('t x', domain='real'); p, pt, px = var('p pt px'); u = var('u'); assume(u > -1, u < 1)
L = pt^2 / 2 - px^2 / 2 - (1 - cos(p))
print("dL/dp =", diff(L, p), "; dL/dpt =", diff(L, pt), "; dL/dpx =", diff(L, px))
def EL(phi):
    s = {p: phi, pt: diff(phi, t), px: diff(phi, x)}
    return diff(diff(L, pt).subs(s), t) + diff(diff(L, px).subs(s), x) - diff(L, p).subs(s)
f = function('f')(t, x)
print("Euler-Lagrange expression:", EL(f))
static = 4 * arctan(exp(x))
# use sin(4 arctan w) = 4 w (1 - w^2)/(1 + w^2)^2, the double-angle formula applied twice with tan(arctan w) = w
w = var('w'); assume(w > 0)
sin4 = 4 * w * (1 - w^2) / (1 + w^2)^2
print("sin(4 arctan w) = 4w(1 - w^2)/(1 + w^2)^2 by the multiple-angle formulas:", bool((sin(4 * arctan(w)).expand_trig() - sin4).simplify_full() == 0))
lhs = (diff(static, x, 2)).simplify_full()
print("static kink: phi'' = sin(phi):", bool((lhs - sin4.subs(w=exp(x))).simplify_full() == 0))
g = 1 / sqrt(1 - u^2)
moving = 4 * arctan(exp(g * (x - u * t)))
lhs = (diff(moving, t, 2) - diff(moving, x, 2)).simplify_full()
print("boosted kink: phi_tt - phi_xx + sin(phi) = 0:", bool((lhs + sin4.subs(w=exp(g * (x - u * t)))).simplify_full() == 0))
print("topological charge (phi(+oo) - phi(-oo))/(2 pi) =", (limit(static, x=oo) - limit(static, x=-oo)) / (2 * pi))
