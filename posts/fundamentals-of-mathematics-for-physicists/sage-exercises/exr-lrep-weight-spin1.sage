# Exercise exr-lrep-weight-spin1: the weights of the vector representation of so(3) = su(2): the eigenvalues and
# eigenvectors of i L_3 with (L_a)_{bc} = -epsilon_{abc}.
L = [matrix(QQbar, 3, 3, lambda b, c: -sign(Permutation([a + 1, b + 1, c + 1])) if len({a, b, c}) == 3 else 0) for a in range(3)]
print("L_3 =", L[2].rows())
print("[L_1, L_2] = L_3:", L[0] * L[1] - L[1] * L[0] == L[2])
H = I * L[2]
for ev, vecs, mult in H.eigenvectors_right():
    print("weight", ev, "multiplicity", mult, "eigenvector", vecs[0])
print("weights are 1, 0, -1 (spin 1):", sorted(H.eigenvalues()) == [-1, 0, 1])
