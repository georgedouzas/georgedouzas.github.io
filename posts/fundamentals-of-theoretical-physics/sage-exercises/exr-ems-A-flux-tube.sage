# Exercise exr-ems-A-flux-tube: A = Phi (-y, x, 0)/(2 pi s^2); curl, circulations, and the multivalued angle.
x, y, z, Phi, R, t = var('x y z Phi R t'); assume(R > 0)
A = Phi / (2 * pi) * vector([-y, x, 0]) / (x^2 + y^2)
curl = vector([diff(A[2], y) - diff(A[1], z), diff(A[0], z) - diff(A[2], x), diff(A[1], x) - diff(A[0], y)]).simplify_full()
print("curl A for s > 0:", curl)
def circ(cx, cy, rad):
    px, py = cx + rad * cos(t), cy + rad * sin(t)
    return integrate((A[0].subs(x=px, y=py) * diff(px, t) + A[1].subs(x=px, y=py) * diff(py, t)).simplify_full(), t, 0, 2 * pi)
print("circulation around the axis:", circ(0, 0, R))
print("circulation on a circle not enclosing the axis (centre (3,0), radius 1):", circ(3, 0, 1).simplify_full())
g = vector([diff(Phi / (2 * pi) * arctan(y / x), v) for v in [x, y, z]])
print("A - grad(Phi arctan(y/x)/(2 pi)) =", (A - g).simplify_full())
