# Exercise exr-cs-reductive-killing-complement: the Killing-orthogonal complement is a reductive complement. For so(3)
# with r = span(L_3) and for su(3) with r = span(Y_3, Y_8) (Y_A = i lambda_A), compute the Killing form
# kappa(X, Y) = tr(ad_X ad_Y), the orthogonal complement k of r, and check [r, k] in k.
def analyse(name, basis, coef, r):
    n = len(basis)
    ad = [matrix(SR, n, n, lambda b, a: coef(basis[A] * basis[a] - basis[a] * basis[A], b)) for A in range(n)]
    kappa = matrix(SR, n, n, lambda A, B: (ad[A] * ad[B]).trace().simplify_full())
    print(name, "Killing form matrix:", kappa.rows())
    comp = [A for A in range(n) if A not in r and all(kappa[A, i] == 0 for i in r)]
    print(name, "basis elements orthogonal to r:", [A + 1 for A in comp], " dim k =", n - len(r))
    ok = all(all(coef(basis[i] * basis[a] - basis[a] * basis[i], j) == 0 for j in r) for i in r for a in comp)
    print(name, "[r, k] in k:", ok)
L = [matrix(SR, [[0,0,0],[0,0,-1],[0,1,0]]), matrix(SR, [[0,0,1],[0,0,0],[-1,0,0]]), matrix(SR, [[0,-1,0],[1,0,0],[0,0,0]])]
analyse("so(3)", L, lambda Z, b: (-(Z * L[b]).trace() / 2).simplify_full(), [2])
lam = [matrix(SR, [[0,1,0],[1,0,0],[0,0,0]]), matrix(SR, [[0,-I,0],[I,0,0],[0,0,0]]), matrix(SR, [[1,0,0],[0,-1,0],[0,0,0]]),
       matrix(SR, [[0,0,1],[0,0,0],[1,0,0]]), matrix(SR, [[0,0,-I],[0,0,0],[I,0,0]]), matrix(SR, [[0,0,0],[0,0,1],[0,1,0]]),
       matrix(SR, [[0,0,0],[0,0,-I],[0,I,0]]), matrix(SR, [[1,0,0],[0,1,0],[0,0,-2]])/sqrt(3)]
Y = [I * l for l in lam]
analyse("su(3)", Y, lambda Z, b: (-(Z * Y[b]).trace() / 2).simplify_full(), [2, 7])
