# Exercise exr-lg-exp-noncommuting: for the non-commuting X = E = [[0,1],[0,0]] and Y = F = [[0,0],[1,0]],
# e^X e^Y != e^{X+Y}; the first correction is the commutator: e^{sX} e^{sY} = e^{s(X+Y) + (s^2/2)[X,Y] + O(s^3)}.
s = var("s")
X = matrix(SR, [[0, 1], [0, 0]]); Y = matrix(SR, [[0, 0], [1, 0]])
print("e^X e^Y =", (X.exp() * Y.exp()).list())
EXY = (X + Y).exp().apply_map(lambda e: e.simplify_full())
print("e^(X+Y) =", EXY.list(), "; equals cosh(1) 1 + sinh(1)(X + Y):", bool(EXY == (cosh(1) * identity_matrix(SR, 2) + sinh(1) * (X + Y)).apply_map(lambda e: e.exponentialize())))
print("(X + Y)^2 =", ((X + Y)^2).list(), "; [X, Y] =", (X * Y - Y * X).list())
lhs = ((s * X).exp() * (s * Y).exp()).apply_map(lambda e: e.taylor(s, 0, 2))
Z = s * (X + Y) + s^2 / 2 * (X * Y - Y * X)
rhs = (identity_matrix(SR, 2) + Z + Z * Z / 2).apply_map(lambda e: e.taylor(s, 0, 2))
naive = (identity_matrix(SR, 2) + s * (X + Y) + s^2 * (X + Y)^2 / 2)
print("e^{sX} e^{sY} to order s^2 =", lhs.list())
print("agrees with exp(s(X+Y) + s^2/2 [X,Y]) to order s^2:", bool(lhs == rhs), "; with exp(s(X+Y)):", bool(lhs == naive))
