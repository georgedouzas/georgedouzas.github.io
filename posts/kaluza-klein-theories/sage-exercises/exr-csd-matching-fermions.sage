# Exercise exr-csd-matching-fermions: the charges of SU(3) representations under the embedded U(1),
# tau = (i/3) diag(1, 1, -2). The weights of an irreducible representation are read from Sage's WeylCharacterRing
# in ambient coordinates (w_1, w_2, w_3), shifted to sum zero; the tau-charge of a weight v is (v_1 + v_2 - 2 v_3)/3. Fermions on S^2 need the spinor charges +-1/2.
A2 = WeylCharacterRing("A2", style="coroots")
reps = {"3": A2(1, 0), "3bar": A2(0, 1), "6": A2(2, 0), "8": A2(1, 1), "10": A2(3, 0), "15": A2(2, 1), "27": A2(2, 2)}
allq = set(); byrep = {}
for name, R in reps.items():
    qs = []
    for w, m in R.weight_multiplicities().items():
        w = [QQ(w[k]) for k in range(3)]; s = sum(w) / 3; v = [x - s for x in w]     # traceless coordinates
        qs += [(v[0] + v[1] - 2 * v[2]) / 3] * m                                  # eigenvalue of diag(1, 1, -2)/3
    qs = sorted(qs); allq |= set(qs); byrep[name] = qs
    print(name, "(dim %d): tau-charges" % R.degree(), qs)
print("[PASS] the 3 has charges 1/3, 1/3, -2/3 and the 8 has 0 (x4), +1 (x2), -1 (x2):",
      byrep["3"] == [QQ(-2)/3, QQ(1)/3, QQ(1)/3] and byrep["8"] == [-1, -1, 0, 0, 0, 0, 1, 1])
print("[PASS] every charge is a multiple of 1/3, so +-1/2 never occurs and no fermion survives on S^2 with this embedding:",
      all((3 * x).denominator() == 1 for x in allq) and QQ(1)/2 not in allq and QQ(-1)/2 not in allq)
