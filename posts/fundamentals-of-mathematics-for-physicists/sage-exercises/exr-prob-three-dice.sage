# Exercise exr-prob-three-dice: the 216 outcomes of three dice; P(sum >= 15), the same given a first six, and the
# distribution of the sum with its modes 10 and 11.
import itertools
Omega = list(itertools.product(range(1, 7), repeat=3))
P = lambda E: QQ(len(E)) / len(Omega)
print("outcomes:", len(Omega))
A = [w for w in Omega if sum(w) >= 15]
B = [w for w in Omega if w[0] == 6]
print("P(sum >= 15) =", P(A))
print("P(sum >= 15 | first = 6) =", P([w for w in A if w in B]) / P(B))
counts = {s: len([w for w in Omega if sum(w) == s]) for s in range(3, 19)}
print("counts of each sum:", counts)
print("most likely sums:", [s for s, c in counts.items() if c == max(counts.values())])
