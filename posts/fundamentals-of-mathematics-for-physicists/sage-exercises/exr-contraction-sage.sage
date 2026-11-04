# Exercise exr-contraction-sage: contractions with SageManifolds' free-module tensors: the trace of a type (1, 1) tensor,
# the contraction of a type (2, 1) tensor T^{ij}_k over i and k (a vector), and the pairing of a covector with a vector.
V = FiniteRankFreeModule(QQ, 3, name="V")
e = V.basis("e")
T = V.tensor((1, 1), name="T"); T[:] = [[1, 2, 0], [0, 3, 1], [0, 0, 5]]
print("trace of T (type (1, 1)):", T.trace(), "= 1 + 3 + 5")
S = V.tensor((2, 1), name="S")
S[0, 0, 0] = 1; S[1, 0, 1] = 2; S[2, 1, 2] = 3; S[0, 2, 0] = 4
c = S.trace(0, 2)   # contract the first (upper) index with the third (lower) one
print("S^{ij}_k contracted over i = k gives the vector with components", c.comp(e)[:], "of type", c.tensor_type())
a = V.linear_form(name="a"); a[:] = [1, 0, -1]; u = V([1, 2, 3], basis=e, name="u")
print("the pairing a_i u^i as a contraction:", a.contract(u), "= a(u) =", a(u))
