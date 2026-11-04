# Exercise exr-qz-hypercharge-anomalies: anomaly sums over one Standard Model generation of left-handed Weyl fermions,
# each listed as (name, multiplicity = colours x weak components, hypercharge Y, is weak doublet, is colour triplet).
F = [("Q", 6, QQ(1) / 6, True, True), ("u^c", 3, -QQ(2) / 3, False, True), ("d^c", 3, QQ(1) / 3, False, True),
     ("L", 2, -QQ(1) / 2, True, False), ("e^c", 1, QQ(1), False, False)]
print("sum Y^3          =", sum(n * Y ** 3 for _, n, Y, _, _ in F))
print("sum Y (gravity)  =", sum(n * Y for _, n, Y, _, _ in F))
print("SU(2)^2 U(1): sum over doublets of Y (per doublet, times colours) =", 3 * QQ(1) / 6 + (-QQ(1) / 2))
print("SU(3)^2 U(1): sum over triplets of Y (per triplet, times weak components) =", 2 * QQ(1) / 6 + (-QQ(2) / 3) + QQ(1) / 3)
quarks = [f for f in F if f[4]]
print("quarks alone: sum Y^3 =", sum(n * Y ** 3 for _, n, Y, _, _ in quarks), ", sum Y =", sum(n * Y for _, n, Y, _, _ in quarks))
