# Exercise exr-sr-find-log: the generator X = log(2) K1 + (pi/2) J1 of B(0.6 x) R_x(pi/2).
K1 = matrix(SR, 4, 4); K1[0, 1] = K1[1, 0] = 1
J1 = matrix(SR, 4, 4); J1[2, 3] = -1; J1[3, 2] = 1
print("[K1, J1] = 0:", K1*J1 == J1*K1)
X = log(2)*K1 + pi/2*J1
E = X.exp().simplify_full()
B = matrix(SR, [[5/4, 3/4, 0, 0], [3/4, 5/4, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
R = matrix(SR, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
print("exp(X) ="); print(E)
print("equals B(0.6 x) R_x(pi/2):", E == B*R)
