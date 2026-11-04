# Exercise exr-vcp-cy-hypersurface: a degree-d hypersurface X in CP^{m+1} has total Chern class (1 + J)^{m+2}/(1 + dJ),
# so c_1 = (m + 2 - d) J and X is Calabi-Yau exactly for d = m + 2: the cubic curve, the quartic surface, the quintic
# threefold and the sextic fourfold. The Euler number is the top Chern class integrated, int J^m = d.
R.<J> = PowerSeriesRing(QQ, default_prec=8)
for m in range(1, 5):
    cy = [d for d in range(1, 10) if ((1 + J) ** (m + 2) / (1 + d * J))[1] == 0]
    d = m + 2
    c = (1 + J) ** (m + 2) / (1 + d * J)
    print("complex dimension %d: c_1 = 0 only for degree %s in CP^%d; c(TX) = %s; chi = %d * %d = %d"
          % (m, cy, m + 1, " + ".join("%s J^%d" % (c[i], i) for i in range(m + 1)), c[m], d, c[m] * d))
print("the cubic curve is a torus (chi = 0), the quartic is K3 (chi = 24), the quintic has chi = -200, the sextic fourfold chi = 2610")
