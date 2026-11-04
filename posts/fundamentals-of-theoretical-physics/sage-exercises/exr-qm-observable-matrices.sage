# Exercise exr-qm-observable-matrices: which matrices are Hermitian, and the spectra of those that are.
s1 = matrix(SR, [[0, 1], [1, 0]]); s2 = matrix(SR, [[0, -I], [I, 0]])
mats = {"A": matrix(SR, [[1, I], [-I, 2]]), "B": matrix(SR, [[0, 1], [0, 0]]),
        "C": matrix(SR, [[0, -I, 0], [I, 0, -I], [0, I, 0]]), "D": s1 * s2}
for name, M in mats.items():
    herm = M == M.conjugate_transpose()
    print(name, "Hermitian:", herm, (" eigenvalues: %s" % M.eigenvalues()) if herm else (" M = %s" % list(M)))
