# Exercise exr-lrep-branch-6: branching of the 6 of su(3) under su(2) x u(1): its weights are the symmetric sums of two
# weights of the 3; the lambda_8-charge sorts them into 3_{2/sqrt3} + 2_{-1/sqrt3} + 1_{-4/sqrt3}.
from itertools import combinations_with_replacement
w3 = {"u": (1, 1/sqrt(3)), "d": (-1, 1/sqrt(3)), "s": (0, -2/sqrt(3))}      # (lambda_3, lambda_8) weights of the 3
w6 = [((a + b), tuple(SR(x + y).simplify_full() for x, y in zip(w3[a], w3[b]))) for a, b in combinations_with_replacement("uds", 2)]
rows = {}
for name, (l3, l8) in w6:
    rows.setdefault(l8, []).append((name, l3))
for q in sorted(rows, key=lambda q: -float(q)):
    print(f"lambda_8-charge {q}: states {[n for n, _ in rows[q]]}, lambda_3-weights {sorted([l for _, l in rows[q]], reverse=True)} -> su(2) multiplet of dimension {len(rows[q])}")
A2 = WeylCharacterRing("A2", style="coroots"); A1 = WeylCharacterRing("A1", style="coroots")
print("Sage levi branching of A2(2,0) = 6:", A2(2, 0).branch(A1, rule="levi"))
print("sum of the charges vanishes:", bool(sum(l8 for _, (l3, l8) in w6) == 0))
