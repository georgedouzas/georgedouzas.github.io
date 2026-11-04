# Exercise exr-fan-parallelogram: D(p) = |u+v|_p^2 + |u-v|_p^2 - 2|u|_p^2 - 2|v|_p^2 for two pairs of vectors in R^2.
p = var('p'); assume(p >= 1)
nrm = lambda x, p: (sum(abs(t)^p for t in x))^(1 / p)
def D(u, v):
    w1 = [a + b for a, b in zip(u, v)]; w2 = [a - b for a, b in zip(u, v)]
    return (nrm(w1, p)^2 + nrm(w2, p)^2 - 2 * nrm(u, p)^2 - 2 * nrm(v, p)^2).simplify_full()
D1 = D((1, 0), (0, 1)); D2 = D((1, 1), (1, -1))
print("u = (1,0), v = (0,1): D(p) =", D1, " zero at", solve(D1 == 0, p))
print("u = (1,1), v = (1,-1): D(p) =", D2, " zero at", solve(D2 == 0, p))
print("D at p = 1, 2, 3, 10:", [(D1.subs(p=q), D2.subs(p=q)) for q in (1, 2, 3, 10)])
sup = lambda x: max(abs(t) for t in x)
print("sup norm, first pair:", sup((1, 1))^2 + sup((1, -1))^2 - 2 * sup((1, 0))^2 - 2 * sup((0, 1))^2)
