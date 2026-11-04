# Exercise exr-partials-chain: partial derivatives, the gradient, a directional derivative and the chain rule along a
# curve, with Sage's symbolic calculus.
x, y, t = var("x y t", domain="real")
f = x ** 2 * y + sin(y)
print("df/dx =", diff(f, x), "; df/dy =", diff(f, y))
print("mixed partials agree (smooth function):", bool(diff(f, x, y) == diff(f, y, x)))
grad = vector([diff(f, x), diff(f, y)])
print("directional derivative at (1, 0) along (3, 4)/5:", (grad.subs(x=1, y=0) * vector([3/5, 4/5])))
c = (cos(t), sin(t))
lhs = diff(f.subs(x=c[0], y=c[1]), t)
rhs = grad.subs(x=c[0], y=c[1]) * vector([diff(c[0], t), diff(c[1], t)])
print("chain rule d/dt f(c(t)) = grad f . c'(t):", bool((lhs - rhs).simplify_full() == 0))
print("Taylor expansion of f around (0, 0) to order 3:", f.taylor((x, 0), (y, 0), 3))
