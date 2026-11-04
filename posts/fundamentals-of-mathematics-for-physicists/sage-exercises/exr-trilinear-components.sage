# Exercise exr-trilinear-components: the components T_ijk = T(e_i, e_j, e_k) of the trilinear map T(u, v, w) = u1 v2 w3
# - u2 v1 w3 on Q^3 (27 components, two non-zero), and the value T(u, v, w) = T_ijk u^i v^j w^k on three vectors.
V = FiniteRankFreeModule(QQ, 3, name="V")
e = V.basis("e")
T = V.tensor((0, 3), name="T")
T[0, 1, 2] = 1; T[1, 0, 2] = -1
print("non-zero components T_ijk (indices from 0):", {k: v for k, v in T.comp(e)._comp.items()}, "; number of components: 3^3 =", 3 ** 3)
u = V([1, 2, 3], basis=e); v = V([0, 1, -1], basis=e); w = V([2, 0, 1], basis=e)
print("T(u, v, w) =", T(u, v, w), "; by the formula u1 v2 w3 - u2 v1 w3 =", 1 * 1 * 1 - 2 * 0 * 1)
print("linearity in the first slot: T(2u + v, v, w) = 2 T(u, v, w) + T(v, v, w):", T(2 * u + v, v, w) == 2 * T(u, v, w) + T(v, v, w))
