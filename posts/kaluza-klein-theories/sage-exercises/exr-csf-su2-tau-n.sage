# Exercise exr-csf-su2-tau-n: SU(2) gauge group, embedding tau_n(Q_3) = n Q_3, fermion in the spin-j representation; the
# charges are -n m, m = -j..j, and a component survives the fermion constraint when its charge is +-1/2.
for n in range(1, 5):
    row = []
    for twoj in range(0, 7):
        j = QQ(twoj) / 2
        ms = [-j + k for k in range(twoj + 1)]
        row.append((j, len([m for m in ms if abs(n * m) == QQ(1) / 2])))
    print("n = %d: (j, surviving components) =" % n, row)
