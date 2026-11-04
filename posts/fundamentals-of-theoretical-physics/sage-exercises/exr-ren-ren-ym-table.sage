# Exercise exr-ren-ren-ym-table: the superficial degree omega = 4 - E_A - 3 E_psi/2 + (D - 4) L of gauge theories in D
# dimensions, checked against direct counting for the one-loop two- and three-gluon graphs, and tabulated.
omega = lambda D, EA, Epsi, L: 4 - EA - 3 * Epsi / 2 + (D - 4) * L
# direct counting: one-loop bubble of two cubic vertices (I = 2, two derivatives), triangle of three (I = 3, three derivatives)
direct = lambda D, I, delta: D * 1 - 2 * I + delta
for D in [4, 5, 6, 10]:
    assert omega(D, 2, 0, 1) == direct(D, 2, 2) and omega(D, 3, 0, 1) == direct(D, 3, 3)
print("formula agrees with direct counting for the one-loop bubble and triangle in D = 4, 5, 6, 10")
print("D   amplitude    L=1  L=2  L=3")
for D in [4, 5, 6, 10]:
    for EA in [2, 3]:
        print("%-3d %d gluons     %4s %4s %4s" % ((D, EA) + tuple(omega(D, EA, 0, L) for L in [1, 2, 3])))
