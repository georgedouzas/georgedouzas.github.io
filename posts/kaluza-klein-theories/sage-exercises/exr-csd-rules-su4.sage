# Exercise exr-csd-rules-su4: the CSDR rules for G = SU(4) over S^2 with tau(Q_3) = (i/2) diag(1, 1, -1, -1).
# The charges t_i - t_j of the matrix units under ad tau, the centraliser and the surviving scalars.
from collections import Counter
t = [QQ(1)/2, QQ(1)/2, QQ(-1)/2, QQ(-1)/2]
units = [(i, j) for i in range(4) for j in range(4) if i != j]
ch = {u: t[u[0]] - t[u[1]] for u in units}
cnt = Counter(ch.values()); cnt[0] += 3                 # the three Cartan directions of su(4)
print("charges of the 15 states of the adjoint of SU(4):", dict(cnt))
print("[PASS] 15 -> 7 states of charge 0, 4 of charge +1, 4 of charge -1:", cnt == Counter({0: 7, 1: 4, -1: 4}))
plus = sorted(u for u in units if ch[u] == 1)
print("charge +1 matrix units (the scalar phi_+):", [(i + 1, j + 1) for i, j in plus])
print("[PASS] the charge +1 units fill the upper-right 2 x 2 block: a (2, 2) bi-doublet of SU(2) x SU(2), 4 complex = 8 real scalars:",
      plus == [(0, 2), (0, 3), (1, 2), (1, 3)])
zero = sorted(u for u in units if ch[u] == 0)
print("[PASS] the charge-0 units are the two diagonal 2 x 2 blocks: H = S(U(2) x U(2)), of dimension 3 + 3 + 1 = 7:", zero == [(0, 1), (1, 0), (2, 3), (3, 2)] and 4 + 3 == 7)
print("[PASS] no charge +-2 states, so tau is the right normalisation (t_1 - t_4 = 1):", all(abs(c) <= 1 for c in ch.values()))
