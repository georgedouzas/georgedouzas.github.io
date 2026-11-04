# Exercise exr-vcr-twist-sage-modes: the Scherk-Schwarz ansatz phi(x, y) = e^{i omega y} phi0(x) on a circle of radius R.
# Check the monodromy, the y-independence of |d_y phi|^2, and the mass operator on the twisted modes.
y, R, om, n = var("y R omega n"); assume(R > 0)
p0r, p0i = var("p0r p0i", domain="real")
phi = exp(I * om * y) * (p0r + I * p0i)
print("monodromy phi(y + 2 pi R)/phi(y) =", (phi.subs(y=y + 2 * pi * R) / phi).simplify_full())
dy = diff(phi, y)
print("|d_y phi|^2 =", (dy * dy.conjugate()).simplify_full().subs(om=om), " (independent of y)")
f = exp(I * (n / R + om) * y)
print("-f''/f for the mode n:", (-diff(f, y, 2) / f).simplify_full(), "; periodic part e^{i n y/R} has period 2 pi R for integer n")
