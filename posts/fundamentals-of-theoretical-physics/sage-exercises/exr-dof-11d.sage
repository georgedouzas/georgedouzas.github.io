# Exercise exr-dof-11d: count the physical degrees of freedom of the supergravity multiplets in d dimensions with
# the light-cone formulas of Tanii (hep-th/9802138, eq. 4.8) and check the boson-fermion matching for the maximal
# theories in d = 11, 10, 5 and 4 (Tanii, Table 3).
def graviton(d): return (d - 2) * (d - 1) / 2 - 1
def pform(d, p): return binomial(d - 2, p)
def gravitino(d, majorana_weyl=False): return (d - 3) * 2^(d // 2) / 2 / (2 if majorana_weyl else 1)
def spinhalf(d, majorana_weyl=False): return 2^(d // 2) / 2 / (2 if majorana_weyl else 1)
d11 = (graviton(11) + pform(11, 3), gravitino(11))
print("d = 11: bosons = 44 + 84 =", d11[0], "; fermions =", d11[1])
print("[PASS] d = 11: 128 + 128:", d11 == (128, 128))
# d = 10 type IIA: graviton, 3-form, 2-form, vector, scalar; two Majorana-Weyl gravitini and two MW spin-1/2
iia_b = graviton(10) + pform(10, 3) + pform(10, 2) + pform(10, 1) + 1
iia_f = 2 * gravitino(10, True) + 2 * spinhalf(10, True)
print("d = 10 (1,1): bosons =", iia_b, "; fermions =", iia_f)
print("[PASS] d = 10 type IIA: 128 + 128:", (iia_b, iia_f) == (128, 128))
# d = 5, N = 8: graviton, 27 vectors, 42 scalars; 8 symplectic-Majorana gravitini and 48 spin-1/2 (each half a Dirac spinor,
# which is what Tanii's formulas count)
d5_b = graviton(5) + 27 * pform(5, 1) + 42; d5_f = 8 * gravitino(5) + 48 * spinhalf(5)
print("d = 5 N = 8: bosons =", d5_b, "; fermions =", d5_f)
print("[PASS] d = 5, N = 8: 128 + 128:", (d5_b, d5_f) == (128, 128))
# d = 4, N = 8: graviton, 28 vectors, 70 scalars; 8 Majorana gravitini, 56 Majorana spin-1/2
d4_b = graviton(4) + 28 * pform(4, 1) + 70; d4_f = 8 * gravitino(4) + 56 * spinhalf(4)
print("d = 4 N = 8: bosons =", d4_b, "; fermions =", d4_f)
print("[PASS] d = 4, N = 8: 128 + 128:", (d4_b, d4_f) == (128, 128))
# the 3-form on T^7 gives 35 scalars and 21 vectors, the metric gives 7 vectors and 28 scalars: 28 vectors, 63 + 7 dualised = 70 scalars
print("[PASS] reduction of the 3-form and the metric on T^7: 21 + 7 = 28 vectors, 35 + 28 + 7 (dualised 2-forms) = 70 scalars:",
      binomial(7, 2) + 7 == 28 and binomial(7, 3) + 28 + 7 == 70)
