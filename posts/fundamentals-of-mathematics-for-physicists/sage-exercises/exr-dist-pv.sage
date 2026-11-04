# Exercise exr-dist-pv: principal values as symmetric limits: PV int_{-1}^{3} dx/x = log 3, PV int dx/(x(x^2+1)) = 0,
# PV int cos x/(x - 1) dx = -pi sin 1.
x, eps = var("x epsilon")
assume(eps > 0); assume(eps < 1)
pv1 = limit(integrate(1 / x, x, -1, -eps) + integrate(1 / x, x, eps, 3), epsilon=0)
print("PV int_{-1}^{3} dx/x =", pv1)
f = 1 / (x * (x ** 2 + 1))
pv2 = limit(integrate(f, x, -oo, -eps) + integrate(f, x, eps, oo), epsilon=0)
print("PV int dx/(x(x^2 + 1)) =", pv2)
y = var("y")
print("with x = y + 1: cos x/(x - 1) =", (cos(y + 1) / y).expand(), "; cos(y)/y is odd, so its principal value over R is 0:", bool((cos(-y) / (-y) + cos(y) / y).simplify_full() == 0))
pv3 = -sin(1) * integrate(sin(y) / y, y, -oo, oo)
print("PV int cos x/(x - 1) dx = -sin(1) int sin y/y dy =", pv3, "= -pi sin 1; numerically", pv3.n(digits=8))
