# Exercise exr-zn-characters: the one-dimensional representations (characters) of the cyclic group Z_6 with Sage's
# character table: six characters k -> exp(2 pi i j k / 6), all irreducible representations of an abelian group.
Z6 = CyclicPermutationGroup(6)
T = Z6.character_table()
print("character table of Z_6 (rows: irreducible representations, columns: conjugacy classes):")
print(T)
print("number of irreducible representations:", T.nrows(), "= order of the group:", Z6.order())
print("all irreducible representations are one-dimensional:", all(T[i, 0] == 1 for i in range(T.nrows())))
z = exp(2 * pi * I / 6)
print("the character with j = 1 is k -> exp(2 pi i k / 6); its values on k = 0..5:", [z**k for k in range(6)])
