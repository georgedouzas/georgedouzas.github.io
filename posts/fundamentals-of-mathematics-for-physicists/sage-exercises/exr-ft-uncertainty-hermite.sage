# Exercise exr-ft-uncertainty-hermite: Delta x Delta k for the Hermite functions x e^{-x^2/2} and (2x^2 - 1) e^{-x^2/2},
# equal to Delta x^2 since each is an eigenfunction of the transform.
x = var("x")
for name, f in [("x e^{-x^2/2}", x * exp(-x ** 2 / 2)), ("(2x^2 - 1) e^{-x^2/2}", (2 * x ** 2 - 1) * exp(-x ** 2 / 2))]:
    dx2 = integrate(x ** 2 * f ** 2, x, -oo, oo) / integrate(f ** 2, x, -oo, oo)
    print(name, ": Delta x^2 =", dx2, "; Delta k = Delta x (eigenfunction), so Delta x Delta k =", dx2, "; exceeds 1/2:", bool(dx2 > 1 / 2))
