# Exercise exr-ssg-levi-signatures: the contraction eps_{m a_1..a_{n-1}} eps_n^{a_1..a_{n-1}} = (n-1)! s g_mn and the full
# contraction n! s, with s = sign(det g), for constant diagonal metrics: Euclidean and Lorentzian in four dimensions,
# and the negative-definite internal metric g = -delta of the seven-dimensional factor in our signature.
from itertools import permutations
def contractions(g):
    n = len(g)
    perms = list(permutations(range(n)))
    sign = {p: Permutation([i + 1 for i in p]).sign() for p in perms}
    one = [[0] * n for _ in range(n)]
    for p in perms:                       # eps_{p0 p1 ... } eps_{q0 p1 ...} with the same tail forces q = p
        w = prod(1 / g[i] for i in p[1:])  # raise the n-1 contracted indices with the inverse (diagonal) metric
        one[p[0]][p[0]] += sign[p]^2 * w
    full = sum(prod(1 / g[i] for i in p) for p in perms)
    return one, full
for name, g in [("Euclidean 4d", [1, 1, 1, 1]), ("Lorentzian 4d", [1, -1, -1, -1]), ("internal 7d, g = -delta", [-1] * 7)]:
    one, full = contractions(g)
    n = len(g); s = sign(prod(g))
    print(name, ": diagonal of eps eps =", [one[i][i] for i in range(n)], "; full contraction =", full)
    print("[PASS] %s: eps_{m...} eps_n^{...} = (n-1)! sign(det g) g_mn and the full contraction is n! sign(det g):" % name,
          all(one[i][i] == factorial(n - 1) * s * g[i] for i in range(n)) and full == factorial(n) * s)
