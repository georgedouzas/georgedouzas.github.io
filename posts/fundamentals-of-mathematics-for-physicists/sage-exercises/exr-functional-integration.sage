# Exercise exr-functional-integration: the integral over the circle is a linear functional on the Fourier modes; it
# sends 1 to 2 pi and every cos(n phi), sin(n phi) with n >= 1 to 0, so its kernel contains all non-constant modes.
phi = var("phi")
I = lambda f: integrate(f, phi, 0, 2 * pi)
print("I(1) =", I(1), "; I(cos phi) =", I(cos(phi)), "; I(sin 2phi) =", I(sin(2 * phi)), "; I(cos^2 phi) =", I(cos(phi) ^ 2))
f, g = 3 + 2 * cos(phi), 1 - sin(2 * phi)
print("linearity: I(f + g) = I(f) + I(g):", bool(I(f + g) == I(f) + I(g)), "; I(5 f) = 5 I(f):", bool(I(5 * f) == 5 * I(f)))
print("row of I on the basis (1, cos phi, sin phi):", [I(b) for b in [1, cos(phi), sin(phi)]])
