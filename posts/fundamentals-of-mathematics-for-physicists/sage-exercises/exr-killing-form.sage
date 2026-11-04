# Exercise exr-killing-form: the Killing form of so(3) and of su(2) in the bases L_a and T_a.
L = [matrix(QQ, [[0,0,0],[0,0,-1],[0,1,0]]), matrix(QQ, [[0,0,1],[0,0,0],[-1,0,0]]), matrix(QQ, [[0,-1,0],[1,0,0],[0,0,0]])]
def ad(X, basis):
    """Matrix of ad_X = [X, .] in the given basis (coefficients solved by linear algebra)."""
    cols = []
    for Y in basis:
        Z = X*Y - Y*X
        cols.append([ (Z.list()[idx] ) for idx in [7, 2, 3] ])   # coefficients: Z = c0 L0 + c1 L1 + c2 L2 read off entries (2,1),(0,2),(1,0)
    return matrix(QQ, cols).transpose()
K = matrix(QQ, 3, 3, lambda a, b: (ad(L[a], L)*ad(L[b], L)).trace())
print("Killing form in the basis L_a:"); print(K)
print("K(L_a, L_b) = -2 delta_ab:", bool(K == -2*identity_matrix(QQ, 3)))
print("ad L0 ="); print(ad(L[0], L))
