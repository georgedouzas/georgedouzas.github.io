# Exercise exr-clm-projectile: the projectile x'' = 0, z'' = -g solved with desolve; the range and the best angle.
t, g, v0, al = var("t g v0 alpha", domain="positive")
x, z = function("x")(t), function("z")(t)
xs = desolve(diff(x, t, 2) == 0, x, ics=[0, 0, v0 * cos(al)], ivar=t)
zs = desolve(diff(z, t, 2) == -g, z, ics=[0, 0, v0 * sin(al)], ivar=t)
print("x(t) =", xs, "; z(t) =", zs.expand())
tland = [s.rhs() for s in solve(zs == 0, t) if s.rhs() != 0][0]
R = xs.subs(t=tland).trig_reduce()
print("landing time:", tland, "; range:", R)
print("range maximal at alpha = pi/4:", bool(diff(R, al).subs({al: pi / 4}) == 0), "; maximal range:", R.subs({al: pi / 4}))
