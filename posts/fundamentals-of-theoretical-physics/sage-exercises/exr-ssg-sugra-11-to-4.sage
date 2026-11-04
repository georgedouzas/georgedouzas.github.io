# Exercise exr-ssg-sugra-11-to-4 (integrative): eleven-dimensional supergravity on T^7 against the N = 8 helicity ladder.
# Bosons: the 11d metric gives the 4d metric, 7 vectors g_{mu m}, 28 scalars g_mn; the 3-form gives 7 two-forms B_{mu nu m}
# (dual to scalars), 21 vectors B_{mu mn} and 35 scalars B_mnp. Fermions: the 32-component Majorana spinor index splits
# as 8 four-dimensional Majorana spinors (32 = 8 x 4), so psi_mu gives 8 gravitini and psi_m gives 7 x 8 = 56 spin-1/2
# fields. The result is compared with the CPT-completed N = 8 ladder 1, 8, 28, 56, 70 of the first part.
vectors = 7 + binomial(7, 2)
scalars = 7 * 8 / 2 + binomial(7, 3) + 7                                          # 28 metric + 35 three-form + 7 dualised two-forms
gravitini = 32 / 4; spinhalf = 7 * (32 / 4)
print("4d fields from T^7: 1 graviton,", gravitini, "gravitini,", vectors, "vectors,", spinhalf, "spin-1/2,", scalars, "scalars")
def single(N, top):
    lam0 = top - QQ(N) / 2
    return {lam0 + QQ(k) / 2: binomial(N, k) for k in range(N + 1)}
lad = single(8, 2)
print("[PASS] the reduction reproduces the N = 8 ladder: helicity 2:1, 3/2:8, 1:28, 1/2:56, 0:70:",
      (1, gravitini, vectors, spinhalf, scalars) == (lad[2], lad[QQ(3) / 2], lad[1], lad[QQ(1) / 2], lad[0]))
states_b = 2 * 1 + 2 * vectors + scalars; states_f = 2 * gravitini + 2 * spinhalf
print("[PASS] 2 + 56 + 70 = 128 bosonic and 16 + 112 = 128 fermionic states, the 44 + 84 and 128 of eleven dimensions:",
      states_b == 128 and states_f == 128 and 44 + 84 == 128)
print("[PASS] real supercharges: one 32-component Majorana spinor = 8 four-dimensional Majorana spinors of 4 components, N = 8:", 32 / 4 == 8)
