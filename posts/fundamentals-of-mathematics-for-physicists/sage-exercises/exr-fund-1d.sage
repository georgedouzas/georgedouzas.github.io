# Exercise exr-fund-1d: G = |x|/2 is the fundamental solution of d^2/dx^2: G'' = 0 for x != 0, the slope jumps by 1 at
# the origin, and the convolution u = int G(x - y) e^(-y^2) dy satisfies u'' = e^(-x^2).
x, y = var("x y")
Gp, Gm = x / 2, -x / 2                         # G for x > 0 and x < 0
print("G'' away from the origin:", diff(Gp, x, 2), diff(Gm, x, 2))
print("slope jump G'(0+) - G'(0-) =", diff(Gp, x) - diff(Gm, x))
f = exp(-y ** 2)
u = integrate((x - y) / 2 * f, y, -oo, x) + integrate((y - x) / 2 * f, y, x, oo)   # |x - y|/2 split at y = x
u = u.simplify_full()
print("u(x) =", u)
print("u'' = e^(-x^2):", bool((diff(u, x, 2) - exp(-x ** 2)).simplify_full() == 0))
