# Exercise exr-clb-mass-defect (Foulaadvand, Problem 4.7): a chain of N = 101 equal springs (k = 1) and masses m = 1 with
# fixed ends, except the middle mass lambda m. For lambda < 1 a mode appears above the band edge 2, localised at the
# defect; its frequency approaches 2/sqrt(lambda (2 - lambda)), its amplitudes decay as r^|n| with r = lambda/(2 - lambda).
N, c = 101, 50
for lam in (1.0, 0.8, 0.5, 0.25, 2.0):
    K = matrix(RDF, N, N, lambda i, j: 2 if i == j else (-1 if abs(i - j) == 1 else 0))
    Minv = diagonal_matrix(RDF, [1 / lam if i == c else 1 for i in range(N)])
    Msq = diagonal_matrix(RDF, [sqrt(lam) if i == c else 1 for i in range(N)])
    # symmetric form M^(-1/2) K M^(-1/2)
    S = Msq.inverse() * K * Msq.inverse()
    ev, V = S.eigenmatrix_right()
    i = max(range(N), key=lambda j: ev[j, j]); w = sqrt(ev[i, i])
    u = Msq.inverse() * V.column(i); u = u / u.norm()
    pr = 1 / sum(t^4 for t in u)                      # participation ratio: number of masses that move
    line = "lambda = %.2f: highest omega = %.5f" % (lam, w)
    if lam < 1:
        r = lam / (2 - lam)
        line += ", predicted %.5f, |u_(c+1)/u_c| = %.4f vs r = %.4f" % (2 / sqrt(lam * (2 - lam)), abs(u[c + 1] / u[c]), r)
    print(line + ", participation ratio %.1f" % pr)
