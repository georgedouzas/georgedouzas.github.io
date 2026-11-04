# Exercise exr-ssg-e7-coset: the 70 scalars of N = 8 supergravity parametrise E_7(7)/SU(8). Dimensions from Sage's Lie
# algebra data: dim E_7 = 133, dim SU(8) = 63, 133 - 63 = 70; and the 56 of E_7 decomposes under SU(8) as 28 + 28bar,
# the electric and magnetic charges of the 28 vectors. Similar cosets: N = 4, SU(1,1)/U(1) (2 scalars); 5d N = 8,
# E_6(6)/USp(8) (78 - 36 = 42 scalars).
def adjdim(t):
    R = WeylCharacterRing(t, style="coroots"); return R(R.highest_root()).degree()
e7, a7, e6, c4 = adjdim("E7"), adjdim("A7"), adjdim("E6"), adjdim("C4")
print("dim E7 =", e7, ", dim SU(8) =", a7, ", dim E6 =", e6, ", dim USp(8) =", c4)
print("[PASS] dim E_7(7)/SU(8) = 133 - 63 = 70, the scalars of four-dimensional N = 8 supergravity:", e7 == 133 and a7 == 63 and e7 - a7 == 70)
print("[PASS] dim E_6(6)/USp(8) = 78 - 36 = 42, the scalars of five-dimensional N = 8 supergravity:", e6 == 78 and c4 == 36 and e6 - c4 == 42)
E7 = WeylCharacterRing("E7", style="coroots")
dims = sorted(set(E7(w).degree() for w in E7.fundamental_weights()))
print("dimensions of the fundamental representations of E7:", dims)
print("[PASS] the smallest representation of E7 is the 56 = 28 + 28 (electric and magnetic charges of the 28 vectors):", min(dims) == 56 and 56 == 2 * binomial(8, 2))
print("[PASS] the noncompact part 133 - 63 = 70 equals binomial(8, 4), the fourth exterior power of the 8 of SU(8):", e7 - a7 == binomial(8, 4))
