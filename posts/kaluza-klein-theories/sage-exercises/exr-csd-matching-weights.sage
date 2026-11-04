# Exercise exr-csd-matching-weights: the weights of the adjoint of su(3) with respect to the Cartan generators
# T_3 = diag(1, -1, 0)/2 and Y_n = diag(1, 1, -2)/(2 sqrt 3) (Hermitian, normalised Tr = 1/2), the tau-charges
# 2 Y_n / sqrt(3), and the SU(2) multiplets of each charge class.
from collections import defaultdict
T3 = [QQ(1)/2, QQ(-1)/2, 0]; Yn = [1 / (2 * sqrt(3)), 1 / (2 * sqrt(3)), -1 / sqrt(3)]
classes = defaultdict(list)
for i in range(3):
    for j in range(3):
        if i != j:
            w = (T3[i] - T3[j], Yn[i] - Yn[j]); c = (2 * w[1] / sqrt(3)).simplify_full()
            classes[c].append(w[0]); print("E_%d%d: weight (T_3, Y_n) =" % (i + 1, j + 1), w, ", tau-charge", c)
classes[0] += [0, 0]
print({k: sorted(v) for k, v in classes.items()})
print("[PASS] the roots have length 1 and the charges are 0 (T_3 = -1, 0, 0, 1: a triplet and a singlet), +1 (T_3 = +-1/2) and -1 (T_3 = +-1/2):",
      sorted(classes[0]) == [-1, 0, 0, 1] and sorted(classes[1]) == [-QQ(1)/2, QQ(1)/2] and sorted(classes[-1]) == [-QQ(1)/2, QQ(1)/2]
      and all(bool((T3[i] - T3[j])^2 + (Yn[i] - Yn[j])^2 == 1) for i in range(3) for j in range(3) if i != j))
