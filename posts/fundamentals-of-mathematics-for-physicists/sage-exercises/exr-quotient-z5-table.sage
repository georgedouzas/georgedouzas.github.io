# Exercise exr-quotient-z5-table: the addition table of Z/5Z.
R = Zmod(5)
print(matrix(ZZ, 5, 5, lambda i, j: (R(i) + R(j)).lift()))
print("[PASS] every row is a permutation of 0..4 (adding a class is a bijection):", all(sorted((R(i) + R(j)).lift() for j in range(5)) == list(range(5)) for i in range(5)))
