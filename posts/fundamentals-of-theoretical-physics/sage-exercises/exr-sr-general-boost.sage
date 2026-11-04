# Exercise exr-sr-general-boost: the boost to a frame moving with an arbitrary velocity v preserves the Minkowski metric.
var('v1 v2 v3', domain='real')
v = vector([v1, v2, v3]); vv = v*v
g = 1/sqrt(1 - vv)
L = matrix(SR, 4, 4)
L[0, 0] = g
for i in range(3):
    L[0, i + 1] = -g*v[i]; L[i + 1, 0] = -g*v[i]
    for j in range(3):
        L[i + 1, j + 1] = (1 if i == j else 0) + (g - 1)*v[i]*v[j]/vv
eta = diagonal_matrix([1, -1, -1, -1])
print("L^T eta L - eta simplifies to zero:", (L.transpose()*eta*L - eta).simplify_full() == 0)
print("L is symmetric:", L == L.transpose())
w = {v1: 3/13, v2: 4/13, v3: 12/13*0}
print("example v = (3/13, 4/13, 0): first row", [x.subs(w).simplify_full() for x in L.row(0)])
