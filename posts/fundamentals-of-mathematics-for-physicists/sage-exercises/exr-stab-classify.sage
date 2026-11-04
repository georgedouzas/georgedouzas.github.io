# Exercise exr-stab-classify: trace, determinant and eigenvalues of three matrices, with the stability verdict.
mats = {"A1": matrix(QQ, [[-3, 1], [1, -3]]), "A2": matrix(QQ, [[1, 2], [3, 0]]), "A3": matrix(QQ, [[0, 2], [-2, -1]])}
for name, A in mats.items():
    tr, det = A.trace(), A.det()
    ev = A.change_ring(QQbar).eigenvalues()
    stable = tr < 0 and det > 0
    kind = "saddle" if det < 0 else ("node" if tr ** 2 >= 4 * det else "spiral")
    print(name, ": tr =", tr, ", det =", det, ", eigenvalues", ev, "->", ("asymptotically stable " if stable else "unstable ") + kind)
    print("    all real parts negative:", all(e.real() < 0 for e in ev), "| matches the trace-determinant criterion:", all(e.real() < 0 for e in ev) == stable)
