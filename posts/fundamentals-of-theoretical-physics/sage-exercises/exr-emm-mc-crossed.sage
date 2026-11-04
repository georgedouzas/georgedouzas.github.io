# Exercise exr-emm-mc-crossed: crossed fields from the Hamiltonian with phi = -E y, A = (-B y, 0, 0).
var('t'); assume(t > 0); m, q, E, B = var('m q E B', domain='positive')
X, Y, PX, PY = var('X Y PX PY')
H = ((PX + q * B * Y)^2 + PY^2) / (2 * m) - q * E * Y
print('H =', H, '; dH/dX =', diff(H, X), '-> p_x conserved')
# with p_x = 0: m y'' = -dH/dY = -(q B)^2 y/m + q E
y = function('y')(t)
sol = desolve(m * diff(y, t, 2) == -(q * B)^2 * y / m + q * E, y, ics=[0, 0, 0], ivar=t).simplify_full()
print('y(t) =', sol, '; centre of oscillation m E/(q B^2)')
xsol = integrate(q * B * sol.subs(t=var('u')) / m, var('u'), 0, t).simplify_full()
print('x(t) =', xsol)
