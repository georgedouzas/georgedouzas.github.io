# Figure: conditional probability on the sample space of two dice (left) and Bayes' rule as a mosaic of a diagnostic
# test (right). Left: the 36 equally likely outcomes (i, j); the event B = "first die at least 4" (blue rows) has 18
# outcomes, the event A = "sum at least 10" (orange) lies inside B, so P(A | B) = 6/18 = 1/3 while P(A) = 1/6; the
# event C = "sum equals 7" (green frames) has exactly 3 outcomes in B, so P(C | B) = 3/18 = 1/6 = P(C): C and B are
# independent. Right: a population with prevalence 1/10 tested with sensitivity 9/10 and false-positive rate 1/5;
# the positive results (lower band) are mostly false positives, and P(ill | positive) = 1/3.
# Output: figures/conditional_probability.svg.
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conditional_probability.svg")

Omega = [(i, j) for i in range(1, 7) for j in range(1, 7)]
P = lambda E: QQ(len(E)) / len(Omega)
A = [w for w in Omega if w[0] + w[1] >= 10]
B = [w for w in Omega if w[0] >= 4]
C = [w for w in Omega if w[0] + w[1] == 7]
inter = lambda E, F: [w for w in E if w in F]
cond = lambda E, F: P(inter(E, F)) / P(F)

def cell(i, j, **kw):
    return polygon2d([(j - 0.5, i - 0.5), (j + 0.5, i - 0.5), (j + 0.5, i + 0.5), (j - 0.5, i + 0.5)], **kw)

g = Graphics()
for (i, j) in Omega:
    col = "lightsteelblue" if (i, j) in B else "white"
    if (i, j) in A:
        col = "darkorange"
    g += cell(i, j, color=col, edgecolor="gray", thickness=0.5)
    g += text(str(i + j), (j, i), color="black", fontsize=8)
for (i, j) in C:
    g += cell(i, j, fill=False, color="darkgreen", thickness=2.5)
for k in range(1, 7):
    g += text(str(k), (k, 0.2), color="black", fontsize=9)
    g += text(str(k), (0.2, k), color="black", fontsize=9)
g += text("second die $j$", (3.5, -0.3), color="black", fontsize=10)
g += text("first die $i$", (-0.45, 3.5), color="black", fontsize=10, rotation=90)
g += text(r"$B$: $i \geq 4$ (blue), $A$: $i + j \geq 10$ (orange), $C$: $i + j = 7$ (green)", (3.5, 7.0), color="black", fontsize=9)
g.axes(False)
g.set_aspect_ratio(1)

# Bayes mosaic: prevalence 1/10, sensitivity 9/10, false-positive rate 1/5
prev, sens, fpr = QQ(1) / 10, QQ(9) / 10, QQ(1) / 5
m = Graphics()
m += polygon2d([(0, 0), (prev, 0), (prev, sens), (0, sens)], color="firebrick", alpha=0.85)
m += polygon2d([(0, sens), (prev, sens), (prev, 1), (0, 1)], color="mistyrose")
m += polygon2d([(prev, 0), (1, 0), (1, fpr), (prev, fpr)], color="darkorange", alpha=0.85)
m += polygon2d([(prev, fpr), (1, fpr), (1, 1), (prev, 1)], color="whitesmoke")
m += line([(prev, 0), (prev, 1)], color="black", thickness=1)
m += line([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)], color="black", thickness=1)
m += text("ill\n$1/10$", (prev / 2, 1.07), color="black", fontsize=9)
m += text("healthy, $9/10$", (prev + (1 - prev) / 2, 1.07), color="black", fontsize=9)
m += text("true positives\n$9/100$", (prev / 2, -0.12), color="firebrick", fontsize=8)
m += text("false positives $18/100$", (prev + (1 - prev) / 2, fpr / 2), color="black", fontsize=9)
m += text("negative", (prev + (1 - prev) / 2, (1 + fpr) / 2), color="gray", fontsize=9)
m += text(r"$P(\mathrm{ill} \mid +) = \frac{9/100}{9/100 + 18/100} = \frac{1}{3}$", (0.55, -0.12), color="black", fontsize=10)
m.axes(False)
m.set_aspect_ratio(1)

arr = graphics_array([[g, m]])
arr.matplotlib(figsize=[9, 4.3]).savefig(OUT, bbox_inches="tight")

# exact checks
print("[PASS] two dice: P(sum >= 10) = 1/6, P(first die >= 4) = 1/2 and P(sum >= 10 | first die >= 4) = 1/3:", P(A) == QQ(1) / 6 and P(B) == QQ(1) / 2 and cond(A, B) == QQ(1) / 3)
print("[PASS] sum = 7 is independent of the first die: P(sum = 7 | first die = i) = 1/6 = P(sum = 7) for every i:", all(cond(C, [w for w in Omega if w[0] == i]) == QQ(1) / 6 for i in range(1, 7)) and P(C) == QQ(1) / 6 and P(inter(C, B)) == P(C) * P(B))
print("[PASS] law of total probability: sum_i P(sum >= 10 | first die = i) P(first die = i) = 1/6:", sum(cond(A, [w for w in Omega if w[0] == i]) * QQ(1) / 6 for i in range(1, 7)) == P(A))
post = prev * sens / (prev * sens + (1 - prev) * fpr)
print("[PASS] Bayes: prevalence 1/10, sensitivity 9/10, false-positive rate 1/5 give P(ill | positive) = 1/3:", post == QQ(1) / 3)
post2 = (QQ(1) / 100 * QQ(99) / 100) / (QQ(1) / 100 * QQ(99) / 100 + QQ(99) / 100 * QQ(1) / 20)
print("[PASS] Bayes: prevalence 1/100, sensitivity 99/100, false-positive rate 1/20 give P(ill | positive) = 1/6:", post2 == QQ(1) / 6)
S = Subsets([1, 2, 3])
closed = all((Set([1, 2, 3]) - E) in S for E in S) and all((E | F) in S for E in S for F in S)
print("[PASS] the 2^3 = 8 subsets of {1, 2, 3} are closed under complement and union (a sigma-algebra):", S.cardinality() == 8 and closed)
print("[PASS] inclusion-exclusion on two dice: P(A or B) = P(A) + P(B) - P(A and B) = 1/2 for A = sum >= 10, B = first >= 4:", P([w for w in Omega if w in A or w in B]) == P(A) + P(B) - P(inter(A, B)) and P([w for w in Omega if w in A or w in B]) == QQ(1) / 2)
print("wrote", os.path.relpath(OUT))
