# Exercise exr-vcr-eq-monopole: the SU(2)-invariant U(1) connection of charge p on CP^1 = S^2 of Dolan-Szabo,
# a_p = p (ybar dy - y dybar)/(2(1 + y ybar)), in real coordinates y = x1 + i x2: compute f_p = d a_p and its integral.
P = Manifold(2, "CP1"); X = P.chart("x1 x2"); x1, x2 = X[:]
p = var("p")
y, yb = x1 + I * x2, x1 - I * x2
dy = P.one_form(1, I); dyb = P.one_form(1, -I)
a = p / (2 * (1 + y * yb)) * (yb * dy - y * dyb)
a = P.one_form(*[a[k].expr().simplify_full() for k in range(2)])
print("a_p =", a.display())
f = a.exterior_derivative()
print("f_p = d a_p =", f.display())
r, t = var("r t", domain="positive")
dens = f[0, 1].expr().subs(x1=r * cos(t), x2=r * sin(t)).simplify_full()
tot = integrate(integrate(dens * r, r, 0, oo), t, 0, 2 * pi)
print("integral of f_p over the plane (orientation dx1 ^ dx2) =", tot, "; (i/2 pi) times it =", (I / (2 * pi) * tot).simplify_full())
print("f_p = -p/(1 + y ybar)^2 dy ^ dybar:", bool((f[0, 1].expr() - (-p / (1 + x1 ^ 2 + x2 ^ 2) ^ 2 * (-2 * I))).simplify_full() == 0))
