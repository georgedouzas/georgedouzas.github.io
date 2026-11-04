# Exercise exr-independence-determinant: n vectors of Q^n are linearly independent exactly when the matrix with these
# columns has non-zero determinant; two families in Q^4, with the dependence relation of the dependent one.
V = QQ ^ 4
F = [vector(QQ, [1, 0, 1, 0]), vector(QQ, [0, 1, 0, 1]), vector(QQ, [1, 1, 0, 0]), vector(QQ, [0, 0, 1, -1])]
G = [vector(QQ, [1, 0, 1, 0]), vector(QQ, [0, 1, 0, 1]), vector(QQ, [1, 1, 0, 0]), vector(QQ, [0, 0, 1, 1])]
for name, fam in [("F", F), ("G", G)]:
    d = matrix(QQ, fam).transpose().det()
    print(name + ": determinant", d, "; independent:", V.linear_dependence(fam) == [], "; dependence relations:", V.linear_dependence(fam))
