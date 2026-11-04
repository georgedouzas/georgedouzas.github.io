# Exercise exr-pr-gamblers-ruin: ruin probabilities r_k for N = 6 from the exact linear system of the recursion
# r_k = p r_{k+1} + q r_{k-1}, r_0 = 1, r_N = 0, against the closed forms 1 - k/N (p = 1/2) and
# ((q/p)^k - (q/p)^N)/(1 - (q/p)^N) (p != q).
N = 6
def ruin(p):
    q = 1 - p
    M = matrix(QQ, N + 1, N + 1); b = vector(QQ, N + 1)
    M[0, 0] = 1; b[0] = 1
    M[N, N] = 1; b[N] = 0
    for k in range(1, N):
        M[k, k] = 1; M[k, k + 1] = -p; M[k, k - 1] = -q
    return M.solve_right(b)
for p in [QQ(1) / 2, QQ(2) / 5, QQ(9) / 20]:
    q = 1 - p
    r = ruin(p)
    closed = [1 - QQ(k) / N for k in range(N + 1)] if p == q else [((q / p) ** k - (q / p) ** N) / (1 - (q / p) ** N) for k in range(N + 1)]
    print("p =", p, ": r_k =", list(r))
    print("       closed form agrees:", all(r[k] == closed[k] for k in range(N + 1)), "; r_3 =", numerical_approx(r[3], digits=4))
p = QQ(9) / 20; q = 1 - p
print("p = 9/20, k = 3: ruin probability for N = 6, 20, 100:", [numerical_approx(((q / p) ** 3 - (q / p) ** n) / (1 - (q / p) ** n), digits=4) for n in [6, 20, 100]])
