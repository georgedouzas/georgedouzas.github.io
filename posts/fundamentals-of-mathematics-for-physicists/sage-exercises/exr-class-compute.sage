# Exercise exr-class-compute: discriminants B^2 - AC and coefficient-matrix eigenvalues of four second-order equations,
# including Tricomi's equation y u_xx + u_yy = 0 whose type changes with y.
y = var("y")
def classify(A, B, C):
    d = SR(B ** 2 - A * C).simplify_full()
    M = matrix(SR, [[A, B], [B, C]])
    kind = "hyperbolic" if bool(d > 0) else ("parabolic" if bool(d == 0) else ("elliptic" if bool(d < 0) else "depends on the point"))
    return d, M.eigenvalues(), kind
for name, (A, B, C) in [("u_xx + 4 u_xy + u_yy", (1, 2, 1)), ("u_xx + 2 u_xy + u_yy", (1, 1, 1)), ("u_xx + 2 u_xy + 3 u_yy", (1, 1, 3)), ("y u_xx + u_yy (Tricomi)", (y, 0, 1))]:
    d, ev, kind = classify(A, B, C)
    print(name, ": B^2 - AC =", d, "| eigenvalues", ev, "->", kind)
print("Tricomi: hyperbolic for y < 0, parabolic at y = 0, elliptic for y > 0:", bool((-y).subs(y=-1) > 0) and bool((-y).subs(y=0) == 0) and bool((-y).subs(y=1) < 0))
