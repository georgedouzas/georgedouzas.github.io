# Exercise exr-phase-equilibria: equilibria and Jacobians of the Lotka-Volterra system x' = x - xy, y' = -y + xy,
# and the conserved quantity x - log x + y - log y.
x, y = var("x y")
F = vector([x - x * y, -y + x * y])
eqs = solve([F[0] == 0, F[1] == 0], [x, y], solution_dict=True)
print("equilibria:", [(e[x], e[y]) for e in eqs])
Jac = jacobian(F, [x, y])
for e in eqs:
    print("Jacobian at", (e[x], e[y]), ":", Jac.subs(e).list(), "| eigenvalues:", Jac.subs(e).eigenvalues())
H = x - log(x) + y - log(y)
dH = (diff(H, x) * F[0] + diff(H, y) * F[1]).simplify_full()
print("dH/dt along the flow:", dH, "-> conserved:", bool(dH == 0))
