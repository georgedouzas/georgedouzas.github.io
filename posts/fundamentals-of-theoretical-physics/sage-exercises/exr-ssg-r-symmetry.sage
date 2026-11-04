# Exercise exr-ssg-r-symmetry: the states at level k of a massless N-extended multiplet are a_{i1}^dag ... a_{ik}^dag |lambda_0>,
# antisymmetric in the R-symmetry index i = 1..N: they form the k-th exterior power of the fundamental of SU(N), of
# dimension binomial(N, k). With Sage's Weyl character ring of SU(8) we check that these are the fundamental
# representations and that Lambda^4 of SU(8) (the 70 scalars of N = 8 supergravity) is self-conjugate.
A7 = WeylCharacterRing("A7", style="coroots")
fw = A7.fundamental_weights()
dims = [A7(fw[k]).degree() for k in range(1, 8)]
print("dimensions of the fundamental representations of SU(8):", dims)
print("[PASS] Lambda^k of the 8 has dimension binomial(8, k): 8, 28, 56, 70, 56, 28, 8:", dims == [binomial(8, k) for k in range(1, 8)])
V = A7(fw[1]); L2 = A7(fw[2])
print("[PASS] 8 x 8 = 28 (antisymmetric, the vectors) + 36 (symmetric):", V * V == L2 + A7(2 * fw[1]) and A7(2 * fw[1]).degree() == 36)
print("[PASS] Lambda^4 (the 70 scalars) is self-conjugate, Lambda^k and Lambda^{8-k} are conjugate:",
      all(A7(fw[k]).dual() == A7(fw[8 - k]) for k in range(1, 8)) and A7(fw[4]).dual() == A7(fw[4]))
A3 = WeylCharacterRing("A3", style="coroots"); f3 = A3.fundamental_weights()
print("[PASS] N = 4 super-Yang-Mills: the 4 gauginos and the 6 scalars are the 4 and Lambda^2 = 6 of SU(4), and the 6 is real:",
      A3(f3[1]).degree() == 4 and A3(f3[2]).degree() == 6 and A3(f3[2]).dual() == A3(f3[2]))
