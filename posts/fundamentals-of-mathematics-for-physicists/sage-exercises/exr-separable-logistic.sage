# Exercise exr-separable-logistic: desolve on the logistic equation y' = y(1 - y), the implicit result solved for y,
# and the general solution 1/(1 + C e^(-x)) verified by substitution.
x, C = var("x C")
y = function("y")(x)
implicit = desolve(diff(y, x) == y * (1 - y), y)
print("desolve returns the separated relation:", implicit)
Y = var("Y")
explicit = solve(Y / (Y - 1) == exp(x + C), Y)          # exponentiate both sides: y/(y - 1) = e^(x + C)
print("exponentiated and solved for y:", explicit)
ygen = 1 / (1 + C * exp(-x))
print("residual of y = 1/(1 + C e^(-x)) in y' - y(1 - y):", (diff(ygen, x) - ygen * (1 - ygen)).simplify_full(), "-> zero:", bool((diff(ygen, x) - ygen * (1 - ygen)).simplify_full() == 0))
