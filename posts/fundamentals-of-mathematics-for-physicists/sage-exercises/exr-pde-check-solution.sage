# Exercise exr-pde-check-solution: e^(-t) sin x solves u_t = u_xx, cos(x - 2t) solves u_tt - 4 u_xx = 0, and
# arctan(y/x) solves u_xx + u_yy = 0.
t, x, y = var("t x y")
checks = [("heat u_t - u_xx", exp(-t) * sin(x), lambda u: diff(u, t) - diff(u, x, 2)),
          ("wave u_tt - 4 u_xx", cos(x - 2 * t), lambda u: diff(u, t, 2) - 4 * diff(u, x, 2)),
          ("Laplace u_xx + u_yy", atan(y / x), lambda u: diff(u, x, 2) + diff(u, y, 2))]
for name, u, op in checks:
    r = op(u).simplify_full()
    print(name, "for u =", u, ": residual", r, "-> solution:", bool(r == 0))
