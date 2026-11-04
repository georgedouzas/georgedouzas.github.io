# Exercise exr-lg-sl2-algebra: the Lie algebra sl(2, R) of SL(2, R): the basis H, E, F, its brackets
# [H, E] = 2E, [H, F] = -2F, [E, F] = H, and the one-parameter subgroups e^{tH}, e^{tE}, e^{tF} of determinant one.
t = var("t", domain="real")
H = matrix(SR, [[1, 0], [0, -1]]); E = matrix(SR, [[0, 1], [0, 0]]); F = matrix(SR, [[0, 0], [1, 0]])
def br(X, Y): return X * Y - Y * X
print("[H, E] = 2E:", br(H, E) == 2 * E, "; [H, F] = -2F:", br(H, F) == -2 * F, "; [E, F] = H:", br(E, F) == H)
for name, X in [("H", H), ("E", E), ("F", F)]:
    G = (t * X).exp().simplify_full()
    print(f"exp(t {name}) =", G.list(), "; det =", G.det().simplify_full())
x, y, z = var("x y z")
print("det exp(xH + yE + zF) = exp(tr) = 1 since tr(xH + yE + zF) =", (x * H + y * E + z * F).trace())
