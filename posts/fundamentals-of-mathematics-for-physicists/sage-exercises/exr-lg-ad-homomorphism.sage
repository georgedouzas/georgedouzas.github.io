# Exercise exr-lg-ad-homomorphism: ad_{[X, Y]} = [ad_X, ad_Y] (the Jacobi identity read as a homomorphism property),
# checked on the matrices of ad_H, ad_E, ad_F of sl(2, R) in the basis (H, E, F).
H = matrix(QQ, [[1, 0], [0, -1]]); E = matrix(QQ, [[0, 1], [0, 0]]); F = matrix(QQ, [[0, 0], [1, 0]])
basis = [H, E, F]; names = ["H", "E", "F"]
def br(X, Y): return X * Y - Y * X
Bmat = matrix(QQ, [b.list() for b in basis]).transpose()
def coords(X): return Bmat.solve_right(vector(QQ, X.list()))
def ad(X): return matrix(QQ, [coords(br(X, b)) for b in basis]).transpose()
for n, b in zip(names, basis):
    print(f"ad_{n} ="); print(ad(b))
print("ad_[X,Y] = [ad_X, ad_Y] for all basis pairs:", all(ad(br(X, Y)) == br(ad(X), ad(Y)) for X in basis for Y in basis))
