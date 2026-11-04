# Exercise exr-mef-three-by-three: exp(tA) for the companion matrix of y''' + y' = 0, the derivative identity, and the
# general solution read off the first row.
t = var("t")
A = matrix(QQ, [[0, 1, 0], [0, 0, 1], [0, -1, 0]])
E = (t * A).exp().apply_map(lambda e: e.rectform().simplify_full())
print("eigenvalues of A:", A.change_ring(QQbar).eigenvalues())
print("exp(tA) =")
print(E)
print("d/dt exp(tA) = A exp(tA):", bool((E.derivative(t) - A * E).apply_map(lambda e: e.simplify_full()) == 0))
y0, y1, y2 = var("y0 y1 y2")
y = (E * vector([y0, y1, y2]))[0].simplify_full()
print("y(t) from the first row:", y)
print("y''' + y' = 0:", bool((diff(y, t, 3) + diff(y, t)).simplify_full() == 0))
