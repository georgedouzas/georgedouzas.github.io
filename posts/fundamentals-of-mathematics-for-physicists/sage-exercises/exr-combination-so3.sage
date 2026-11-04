# Exercise exr-combination-so3: every antisymmetric 3 x 3 matrix is a linear combination a L1 + b L2 + c L3 of the three
# generators (L_a)_{bc} = -eps_{abc}; the coefficients are read off the entries.
L1 = matrix(QQ, [[0, 0, 0], [0, 0, -1], [0, 1, 0]])
L2 = matrix(QQ, [[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
L3 = matrix(QQ, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])
X = matrix(QQ, [[0, -5, 2], [5, 0, -7], [-2, 7, 0]])
print("X antisymmetric:", X.transpose() == -X)
a, b, c = X[2, 1], X[0, 2], X[1, 0]
print("coefficients (a, b, c) = (X_32, X_13, X_21) =", (a, b, c), "; a L1 + b L2 + c L3 == X:", a * L1 + b * L2 + c * L3 == X)
