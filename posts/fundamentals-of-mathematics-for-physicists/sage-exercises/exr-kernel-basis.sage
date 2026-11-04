# Exercise exr-kernel-basis: a basis of the kernel of A = [[1, 2, 3], [2, 4, 6]] (the plane x + 2y + 3z = 0), and the
# kernel vectors (1, 1, -1) and (3, 0, -1) written in that basis.
A = matrix(QQ, [[1, 2, 3], [2, 4, 6]])
K = A.right_kernel()
print("kernel basis (echelon form):", K.basis())
for v in [vector(QQ, [1, 1, -1]), vector(QQ, [3, 0, -1])]:
    print(v, ": A v =", A * v, "; coordinates in the kernel basis:", K.coordinates(v))
print("the two vectors (1, 1, -1), (3, 0, -1) form another basis of the kernel:", K.span([vector(QQ, [1, 1, -1]), vector(QQ, [3, 0, -1])]) == K)
