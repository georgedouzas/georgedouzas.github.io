# Exercise exr-su3-roots: the roots of su(3) from the Gell-Mann matrices, the Cartan matrix, and the branching of the
# adjoint under su(2) x u(1) and under u(1) x u(1).
lam = [matrix(SR, [[0,1,0],[1,0,0],[0,0,0]]), matrix(SR, [[0,-I,0],[I,0,0],[0,0,0]]), matrix(SR, [[1,0,0],[0,-1,0],[0,0,0]]),
       matrix(SR, [[0,0,1],[0,0,0],[1,0,0]]), matrix(SR, [[0,0,-I],[0,0,0],[I,0,0]]), matrix(SR, [[0,0,0],[0,0,1],[0,1,0]]),
       matrix(SR, [[0,0,0],[0,0,-I],[0,I,0]]), matrix(SR, [[1,0,0],[0,1,0],[0,0,-2]])/sqrt(3)]
def charge(X, Hc):
    C = Hc*X - X*Hc
    for e in zip(C.list(), X.list()):
        if e[1] != 0: return (e[0]/e[1]).simplify_full()
E = {"E12+": lam[0] + I*lam[1], "E12-": lam[0] - I*lam[1], "E45+": lam[3] + I*lam[4], "E45-": lam[3] - I*lam[4],
     "E67+": lam[5] + I*lam[6], "E67-": lam[5] - I*lam[6]}
roots = {k: (charge(v, lam[2]), charge(v, lam[7])) for k, v in E.items()}
print("roots (lambda_3, lambda_8 charges):", roots)
a1, a2 = vector(roots["E12+"]), vector(roots["E67+"])          # simple roots: alpha_1 = (2,0), alpha_2 = (-1, sqrt3)
print("simple roots alpha_1, alpha_2 =", a1, a2, "; alpha_1 + alpha_2 =", (a1 + a2), "= root of E45+:", bool(vector(roots["E45+"]) == a1 + a2))
cartan = matrix(SR, [[2*(a*b)/(b*b) for b in (a1, a2)] for a in (a1, a2)]).simplify_full()
print("Cartan matrix A_ij = 2 (alpha_i . alpha_j)/(alpha_j . alpha_j):"); print(cartan)
print("branching of 8 under su(2) x u(1)_8: lambda_8 charges of (lambda_1, lambda_2, lambda_3), lambda_8, E45+-, E67+-:",
      [charge(lam[a], lam[7]) if (lam[7]*lam[a] - lam[a]*lam[7]) != 0 else 0 for a in range(3)], 0, roots["E45+"][1], roots["E45-"][1], roots["E67+"][1], roots["E67-"][1])
print("-> 8 = 3_0 + 1_0 + 2_{sqrt3} + 2_{-sqrt3}; under u(1) x u(1): two zero weights and the six roots")
