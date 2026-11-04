# Exercise exr-linear-desolve: y' + y/x = x and y' - 2xy = x by desolve, the integrating factors, and substitution
# checks of the general solutions.
x = var("x")
y = function("y")(x)
for p, q in [(1 / x, x), (-2 * x, x)]:
    sol = desolve(diff(y, x) + p * y == q, y)
    mu = exp(integrate(p, x))
    print("equation y' + (%s) y = %s: solution y =" % (p, q), sol, "| integrating factor mu =", mu)
    print("   mu' = p mu, so (mu y)' = mu (y' + p y):", bool((diff(mu, x) - p * mu).simplify_full() == 0), "| residual of the solution zero:", bool((diff(sol, x) + p * sol - q).simplify_full() == 0))
