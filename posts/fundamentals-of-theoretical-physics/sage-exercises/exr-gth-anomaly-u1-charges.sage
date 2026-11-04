# Exercise exr-gth-anomaly-u1-charges: left-handed Weyl fermions of integer U(1) charges q_1..q_n have no U(1)^3 anomaly
# if sum q^3 = 0 and no mixed gravitational anomaly if sum q = 0. A vector-like set (q and -q in pairs) solves both
# trivially; we search for chiral solutions, sets with no pair q, -q, among charges |q| <= 9.
from itertools import combinations_with_replacement as cwr
def chiral(qs): return all(-q not in qs for q in qs) and 0 not in qs
found = {}
for n in range(2, 6):
    for qs in cwr([q for q in range(-9, 10) if q != 0], n):
        if sum(qs) == 0 and sum(q^3 for q in qs) == 0 and chiral(qs):
            found.setdefault(n, []).append(qs)
for n in sorted(found): print("n = %d: %d chiral anomaly-free sets, e.g. %s" % (n, len(found[n]), found[n][:3]))
print("[PASS] no chiral anomaly-free set exists with 2, 3 or 4 fermions of charges |q| <= 9:", all(n not in found for n in (2, 3, 4)))
print("[PASS] (1, 5, -7, -8, 9) is a chiral anomaly-free set of five fermions:", (-8, -7, 1, 5, 9) in found.get(5, []))
q = (1, 5, -7, -8, 9)
print("    sum q =", sum(q), "  sum q^3 =", sum(x^3 for x in q))
# two charges: q1 + q2 = 0 forces q2 = -q1, a vector-like pair; three charges: q1 + q2 + q3 = 0 gives sum q^3 = 3 q1 q2 q3
a, b = var("a b")
print("[PASS] for three charges summing to zero, sum q^3 = 3 q1 q2 q3, which vanishes only if a charge is zero:",
      bool((a^3 + b^3 + (-a - b)^3 - 3 * a * b * (-a - b)).expand() == 0))
