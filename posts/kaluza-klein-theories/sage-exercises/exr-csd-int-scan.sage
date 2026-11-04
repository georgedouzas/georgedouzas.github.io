# Exercise exr-csd-int-scan: all diagonal embeddings tau = i diag(t_1, t_2, t_3) of U(1) in SU(3), sum t_i = 0, with
# t_i in (1/6)Z and |t_i - t_j| <= 2, up to permutations of the entries. For each: the dimension of the centraliser,
# the number of complex scalars surviving on S^2 (matrix units of charge +1) and their pattern.
from itertools import product
seen = {}
grid = [QQ(k) / 6 for k in range(-12, 13)]
for t1, t2 in product(grid, grid):
    t = (t1, t2, -t1 - t2)
    if max(t) - min(t) > 2 or max(t) == min(t): continue
    key = tuple(sorted(t, reverse=True))
    if key in seen: continue
    plus = [(i + 1, j + 1) for i in range(3) for j in range(3) if i != j and key[i] - key[j] == 1]
    h = 2 + sum(1 for i in range(3) for j in range(3) if i != j and key[i] == key[j])
    seen[key] = (h, plus)
with_scalars = {k: v for k, v in seen.items() if v[1]}
for k, (h, plus) in sorted(with_scalars.items()):
    print("t =", k, ": dim H =", h, ", complex scalars on S^2:", len(plus), plus)
print("[PASS] up to permutations, six embeddings on this grid leave scalars, and only the two with a repeated entry have a four-dimensional H:", len(with_scalars) == 6 and sorted(k for k, v in with_scalars.items() if v[0] == 4) == [(QQ(1)/3, QQ(1)/3, QQ(-2)/3), (QQ(2)/3, QQ(-1)/3, QQ(-1)/3)])
print("[PASS] only t = (1/3, 1/3, -2/3) gives H = U(2) with a doublet; the conjugate t = (2/3, -1/3, -1/3) gives the conjugate doublet:",
      with_scalars[(QQ(1)/3, QQ(1)/3, QQ(-2)/3)] == (4, [(1, 3), (2, 3)]) and with_scalars[(QQ(2)/3, QQ(-1)/3, QQ(-1)/3)] == (4, [(1, 2), (1, 3)]))
print("[PASS] t = (1/2, -1/2, 0) leaves one singlet and t = (1, 0, -1) two singlets, both with H = U(1) x U(1):",
      with_scalars[(QQ(1)/2, 0, QQ(-1)/2)] == (2, [(1, 3)]) and with_scalars[(1, 0, -1)] == (2, [(1, 2), (2, 3)]))
