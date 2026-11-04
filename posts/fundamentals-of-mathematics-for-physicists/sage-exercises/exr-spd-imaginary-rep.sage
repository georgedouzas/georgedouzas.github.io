# Exercise exr-spd-imaginary-rep: purely imaginary gamma matrices of signature (1,3), and the real matrices i*gamma.
s = [identity_matrix(QQbar, 2), matrix(QQbar, [[0, 1], [1, 0]]), matrix(QQbar, [[0, -I], [I, 0]]), matrix(QQbar, [[1, 0], [0, -1]])]
kr = lambda a, b: a.tensor_product(b)
gt = [kr(s[2], s[1]), I * kr(s[1], s[0]), I * kr(s[2], s[2]), I * kr(s[3], s[0])]
eta = diagonal_matrix([1, -1, -1, -1])
print("Clifford relations with eta = diag(1,-1,-1,-1):", all(gt[a] * gt[b] + gt[b] * gt[a] == 2 * eta[a, b] for a in range(4) for b in range(4)))
print("all entries purely imaginary:", all(x.conjugate() == -x for x in gt))
R = [I * x for x in gt]
print("i*gamma real:", all(x.conjugate() == x for x in R), "; squares:", [ (x * x)[0, 0] for x in R ], "= -eta^{mu mu}")
