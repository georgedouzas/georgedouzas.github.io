# Exercise exr-lg-su2-components: for X = x_a T_a in su(2), the components are recovered as x_a = -2 tr(T_a X), the
# square is X^2 = -(|x|^2/4) 1, and det X = |x|^2/4; the trace form -2 tr(XY) is the Euclidean dot product x.y.
x = vector(SR, var("x1 x2 x3", domain="real")); y = vector(SR, var("y1 y2 y3", domain="real"))
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
T = [-I / 2 * sa for sa in s]
X = sum(x[a] * T[a] for a in range(3)); Y = sum(y[a] * T[a] for a in range(3))
print("X =", X.list())
print("-2 tr(T_a X) =", [(-2 * (T[a] * X).trace()).expand() for a in range(3)])
print("X^2 =", (X * X).expand().list(), "; det X =", X.det().expand())
print("-2 tr(XY) =", (-2 * (X * Y).trace()).expand())
print("X is anti-Hermitian and traceless:", bool(X.conjugate_transpose() == -X), X.trace())
