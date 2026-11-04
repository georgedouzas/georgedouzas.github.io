# Exercise exr-fld-kg-lorentz: the Klein-Gordon operator in 1+1 dimensions, signature (+, -), is invariant under a
# boost of rapidity w: if phi'(t, x) = phi(t cosh w - x sinh w, x cosh w - t sinh w) then
# (d_t^2 - d_x^2) phi' equals (d_t^2 - d_x^2) phi evaluated at the boosted point; the Galilean operator d_t - d_x^2 is not.
t, x, w, m = var("t x w m", domain="real")
phi = function("phi")
tp, xp = t * cosh(w) - x * sinh(w), x * cosh(w) - t * sinh(w)
phib = phi(tp, xp)
lhs = diff(phib, t, 2) - diff(phib, x, 2) + m**2 * phib
T, X = var("T X")
rhs = (diff(phi(T, X), T, 2) - diff(phi(T, X), X, 2) + m**2 * phi(T, X)).subs(T=tp, X=xp)
print("(Box + m^2) phi' = [(Box + m^2) phi] at the boosted point:", bool((lhs - rhs).simplify_full() == 0))
heat = diff(phib, t) - diff(phib, x, 2)
heat_rhs = (diff(phi(T, X), T) - diff(phi(T, X), X, 2)).subs(T=tp, X=xp)
print("the heat operator d_t - d_x^2 is not boost invariant:", not bool((heat - heat_rhs).simplify_full() == 0))
print("the boost preserves t^2 - x^2:", bool((tp**2 - xp**2 - (t**2 - x**2)).simplify_full() == 0))
