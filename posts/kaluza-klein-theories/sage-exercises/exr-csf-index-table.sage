# Exercise exr-csf-index-table: for SU(3) representations and tau' = (i/2) diag(1,-1,0), each weight of charge q contributes
# |2q| zero modes of internal chirality sign(q) on S^2 and one CSDR fermion (right-handed for q = 1/2, left-handed for -1/2).
A2 = WeylCharacterRing("A2", style="coroots")
for hw, name in (((1, 0), "3"), ((2, 0), "6"), ((1, 1), "8"), ((3, 0), "10")):
    zp = zm = R_ = L_ = 0
    for w, mult in A2(hw).weight_multiplicities().items():
        h = [QQ(c) for c in w.to_vector()]; q = (h[0] - h[1]) / 2
        if q > 0: zp += mult * 2 * q
        if q < 0: zm += mult * 2 * abs(q)
        if q == QQ(1) / 2: R_ += mult
        if q == -QQ(1) / 2: L_ += mult
    print("%-3s: zero modes of chirality +1: %s, of chirality -1: %s, index %s; CSDR fermions: %d right-handed, %d left-handed" % (name, zp, zm, zp - zm, R_, L_))
