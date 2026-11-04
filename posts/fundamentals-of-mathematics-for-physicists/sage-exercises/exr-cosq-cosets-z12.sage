# Exercise exr-cosq-cosets-z12: the cosets of the subgroups <4> = {0, 4, 8} and <3> = {0, 3, 6, 9} of Z_12, computed
# by hand with Zmod and with Sage's cosets() on the cyclic permutation group of order 12.
R = Zmod(12)
for d in (4, 3):
    H = {R(k) for k in range(0, 12, d)}
    cos = sorted({tuple(sorted(int(a + h) for h in H)) for a in R})
    print("cosets of <%d>: %s ; number %d = 12/|H| = %d" % (d, cos, len(cos), 12 // len(H)))
    print("   they partition Z_12:", sorted(x for c in cos for x in c) == list(range(12)))
C12 = CyclicPermutationGroup(12); g = C12.gen()
H = C12.subgroup([g**4])
print("Sage: C_12 has", len(C12.cosets(H)), "cosets of <g^4>, each of size", {len(c) for c in C12.cosets(H)})
