# Exercise exr-eigen-compute: eigenvalues and eigenvectors of A = [[-1, 1], [1, -1]] and the solution with x(0) = (3, 1)
# as a combination of exponential solutions.
t = var("t")
A = matrix(QQ, [[-1, 1], [1, -1]])
ev = A.eigenvectors_right()
for lam, vecs, mult in ev:
    print("eigenvalue", lam, "with eigenvector", vecs[0])
x0 = vector(QQ, [3, 1])
P = matrix(QQ, [ev[0][1][0], ev[1][1][0]]).transpose()      # columns: the eigenvectors
c = P.solve_right(x0)
print("x(0) = (3, 1) =", c[0], "*", ev[0][1][0], "+", c[1], "*", ev[1][1][0])
x = c[0] * exp(ev[0][0] * t) * ev[0][1][0] + c[1] * exp(ev[1][0] * t) * ev[1][1][0]
print("x(t) =", x)
print("solves x' = A x:", bool((x.derivative(t) - A * x).simplify_full() == 0))
