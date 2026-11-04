# Exercise exr-cs-symmetry-automorphism: the symmetry of a coset. The linear map sigma = +1 on r, -1 on k is a Lie
# algebra automorphism for S^2 = SO(3)/SO(2), where it is conjugation by D = diag(-1, -1, 1) (the rotation by pi about
# the Z-axis); for the flag manifold SU(3)/U(1)xU(1) it fails on the pairs of coset generators whose bracket has a
# component in k.
L = [matrix(QQ, [[0,0,0],[0,0,-1],[0,1,0]]), matrix(QQ, [[0,0,1],[0,0,0],[-1,0,0]]), matrix(QQ, [[0,-1,0],[1,0,0],[0,0,0]])]
D = diagonal_matrix(QQ, [-1, -1, 1])
print("D L_a D^{-1} = (-L_1, -L_2, L_3):", [D * La * D.inverse() for La in L] == [-L[0], -L[1], L[2]])
print("so conjugation by D, an automorphism, is the symmetry sigma of S^2")
lam = [matrix(SR, [[0,1,0],[1,0,0],[0,0,0]]), matrix(SR, [[0,-I,0],[I,0,0],[0,0,0]]), matrix(SR, [[1,0,0],[0,-1,0],[0,0,0]]),
       matrix(SR, [[0,0,1],[0,0,0],[1,0,0]]), matrix(SR, [[0,0,-I],[0,0,0],[I,0,0]]), matrix(SR, [[0,0,0],[0,0,1],[0,1,0]]),
       matrix(SR, [[0,0,0],[0,0,-I],[0,I,0]]), matrix(SR, [[1,0,0],[0,1,0],[0,0,-2]])/sqrt(3)]
r = [2, 7]
eps = [1 if A in r else -1 for A in range(8)]
def sigma(X):
    c = [((X * lam[A]).trace() / 2).simplify_full() for A in range(8)]
    return sum(eps[A] * c[A] * lam[A] for A in range(8))
br = lambda X, Y: X * Y - Y * X
bad = [(A + 1, B + 1) for A in range(8) for B in range(A + 1, 8)
       if not bool((sigma(br(lam[A], lam[B])) - br(sigma(lam[A]), sigma(lam[B]))).simplify_full() == 0)]
print("su(3), r = span(lambda_3, lambda_8): pairs (A, B) with sigma[l_A, l_B] != [sigma l_A, sigma l_B]:", bad)
print("number of failing pairs:", len(bad), " -> sigma is not an automorphism: the flag manifold is not symmetric")
