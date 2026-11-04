# Exercise exr-lg-sl2-dimension: SL(2, R) = {det = 1} is the level set of a function on R^4 whose gradient never
# vanishes on it, so it is a closed submanifold of dimension 4 - 1 = 3; its tangent space at 1 is the traceless matrices.
a, b, c, d = var("a b c d", domain="real")
f = a * d - b * c
grad = vector([diff(f, v) for v in (a, b, c, d)])
print("grad det =", grad)
print("grad det = 0 forces a = b = c = d = 0, where det = 0, so 1 is a regular value:", solve(list(grad), [a, b, c, d]))
print("gradient at the identity (a, b, c, d) = (1, 0, 0, 1):", grad.subs(a=1, b=0, c=0, d=1))
print("tangent vectors X at 1 satisfy grad . X = X_11 + X_22 = tr X = 0: dimension", 4 - matrix(QQ, [[1, 0, 0, 1]]).rank())
t = var("t"); M = matrix(SR, [[1, t], [0, 1]]) * matrix(SR, [[cos(t), -sin(t)], [sin(t), cos(t)]])
print("a curve in SL(2, R):", M.list(), "; det =", M.det().simplify_full(), "; velocity at t = 0:", diff(M, t).subs(t=0).list(), "trace", diff(M, t).subs(t=0).trace())
