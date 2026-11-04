# Exercise exr-sp-resolvent-2x2: the resolvents of [[2, 1], [0, 1]] and of sigma_1 as symbolic matrices in lambda, the
# resolvent identity, and the growth of ||R(2 + epsilon)|| as epsilon -> 0.
lam, mu = var("lambda_ mu")
A = matrix(SR, [[2, 1], [0, 1]]); s1 = matrix(SR, [[0, 1], [1, 0]])
R = lambda M, z: (M - z * identity_matrix(2)).inverse()
print("R_A(lambda) ="); print(R(A, lam).simplify_full())
print("R_sigma1(lambda) ="); print(R(s1, lam).simplify_full())
print("resolvent identity R(lambda) - R(mu) = (lambda - mu) R(lambda) R(mu):", bool((R(A, lam) - R(A, mu) - (lam - mu) * R(A, lam) * R(A, mu)).simplify_full() == 0))
for eps in [QQ(1) / 10, QQ(1) / 100, QQ(1) / 1000]:
    print("||R(2 + %s)|| =" % eps, R(A, 2 + eps).change_ring(QQbar).norm(2), " ~ 1/epsilon =", 1 / eps)
