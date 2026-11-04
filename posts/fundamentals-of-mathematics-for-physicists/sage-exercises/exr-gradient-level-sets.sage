# Exercise exr-gradient-level-sets: the gradient of f(x, y) = x^2 + 4 y^2 is orthogonal to the level curves (ellipses)
# and points towards increasing f; at (2, 1) the tangent direction of the level curve is orthogonal to grad f.
x, y, t = var("x y t")
f = x ** 2 + 4 * y ** 2
grad = vector([diff(f, x), diff(f, y)])
print("grad f =", grad, "; at (2, 1):", grad.subs(x=2, y=1))
# the level curve f = 8 through (2, 1): parametrise x = sqrt(8) cos t, y = sqrt(2) sin t
c = vector([sqrt(8) * cos(t), sqrt(2) * sin(t)])
print("f on the level curve is constant:", f.subs(x=c[0], y=c[1]).simplify_trig())
tangent = diff(c, t)
dotp = (grad.subs(x=c[0], y=c[1]) * tangent).simplify_trig()
print("grad f . tangent along the level curve =", dotp, "(orthogonal everywhere)")
print("directional derivative at (2, 1) along the unit vector (3, 4)/5:", grad.subs(x=2, y=1) * vector([QQ(3) / 5, QQ(4) / 5]), "; maximal along grad f/|grad f|:", (grad.subs(x=2, y=1).norm()).simplify_full())
