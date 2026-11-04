# Exercise exr-cosq-quotient-z12: the quotient groups Z_12/<4> and Z_12/<6>, by hand (cosets of residues) and with
# Sage's quotient() of the cyclic permutation group.
R = Zmod(12)
def coset(a, d):
    return frozenset(R(a) + R(k) for k in range(0, 12, d))
for d in (4, 6):
    cos = sorted({coset(a, d) for a in range(12)}, key=lambda c: min(int(x) for x in c))
    rep = [min(int(x) for x in c) for c in cos]
    well = all(coset(a + b, d) == coset(a2 + b2, d) for a in range(12) for b in range(12)
               for a2 in range(12) for b2 in range(12) if coset(a, d) == coset(a2, d) and coset(b, d) == coset(b2, d))
    print("Z_12/<%d>: %d cosets with representatives %s ; product well defined: %s" % (d, len(cos), rep, well))
    print("   the coset of 1 has order", min(n for n in range(1, 13) if coset(n, d) == coset(0, d)), "in the quotient")
C12 = CyclicPermutationGroup(12); g = C12.gen()
for d in (4, 6):
    Q = C12.quotient(C12.subgroup([g**d]))
    print("Sage: C_12/<g^%d> has order %d, cyclic: %s" % (d, Q.order(), Q.is_cyclic()))
