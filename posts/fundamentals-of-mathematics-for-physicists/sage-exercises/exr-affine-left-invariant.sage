# Exercise exr-affine-left-invariant: the affine group of the line as 2 x 2 matrices [[a, b], [0, 1]] in Sage: its
# multiplication, its Lie algebra (matrices [[x, y], [0, 0]]), the bracket [X, Y] = Y, and the exponential of a
# generator with Sage's matrix exponential.
a, b, a0, b0, t = var("a b a0 b0 t", domain="real")
M = lambda a, b: matrix(SR, [[a, b], [0, 1]])
print("(a0, b0) * (a, b) =", (M(a0, b0) * M(a, b)).list(), " -> (a0 a, a0 b + b0)")
X = matrix(SR, [[1, 0], [0, 0]]); Y = matrix(SR, [[0, 1], [0, 0]])
print("[X, Y] =", (X*Y - Y*X).list(), "= Y:", X*Y - Y*X == Y)
print("exp(t X) =", (t*X).exp().simplify_full().list(), "; exp(t Y) =", (t*Y).exp().list(), " (the one-parameter subgroups of dilations and translations)")
print("the Lie algebra is two-dimensional and not abelian; the left-invariant fields at (a, b) are a d/da and a d/db")
