# Exercise exr-qm-osc-3d-degeneracy: degeneracies of the isotropic 3D oscillator and their angular momentum content.
for n in range(9):
    count = len([1 for a in range(n + 1) for b in range(n + 1 - a)])
    ls = list(range(n, -1, -2))
    print("n =", n, " states =", count, " (n+1)(n+2)/2 =", (n + 1) * (n + 2) // 2, " l =", ls, " sum (2l+1) =", sum(2 * l + 1 for l in ls))
