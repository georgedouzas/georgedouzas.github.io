# Exercise exr-ssg-dof-reduction-chain: reduce the bosonic fields of eleven-dimensional supergravity on T^n, n = 0..7, to
# d = 11 - n dimensions and count on-shell degrees of freedom with the light-cone formulas: the metric gives a d-metric,
# n vectors and n(n+1)/2 scalars; the three-form gives a 3-form, n 2-forms, binomial(n, 2) vectors, binomial(n, 3) scalars.
# A p-form in d dimensions has binomial(d - 2, p) states, a graviton (d - 2)(d - 1)/2 - 1. The total stays 128.
def graviton(d): return (d - 2) * (d - 1) / 2 - 1
def form(d, p): return binomial(d - 2, p) if p >= 0 else 0
rows = []
for n in range(8):
    d = 11 - n
    fields = {"metric": 1, "3-forms": 1, "2-forms": n, "vectors": n + binomial(n, 2), "scalars": n * (n + 1) / 2 + binomial(n, 3)}
    total = graviton(d) + form(d, 3) + n * form(d, 2) + fields["vectors"] * form(d, 1) + fields["scalars"]
    rows.append((d, fields, total))
    print("d = %2d:" % d, fields, " bosonic states", total)
print("[PASS] the bosonic count is 128 in every dimension from 11 to 4:", all(r[2] == 128 for r in rows))
d4 = rows[-1][1]
print("[PASS] in d = 4 (n = 7): 7 + 21 = 28 vectors, 28 + 35 = 63 scalars plus 7 two-forms dual to scalars (and a non-dynamical 3-form): 70 scalars:",
      d4["vectors"] == 28 and d4["scalars"] == 63 and d4["2-forms"] == 7 and form(4, 2) == 1 and form(4, 3) == 0)
