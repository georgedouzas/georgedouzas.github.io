# Exercise exr-ren-bf-sm: one-loop coefficients b_i = -11/3 C2(G_i) + 2/3 sum_F T_i(F) d'(F) + 1/3 sum_S T_i(S) d'(S) of the
# Standard Model with three generations of Weyl fermions and one Higgs doublet; T_1 = (3/5) Y^2.
fermions = [("Q", 3, 2, 1/6), ("uc", 3, 1, -2/3), ("dc", 3, 1, 1/3), ("L", 1, 2, -1/2), ("ec", 1, 1, 1)]
scalars = [("H", 1, 2, 1/2)]
def contrib(fields, weight):
    b = [0, 0, 0]
    for name, n3, n2, Y in fields:
        b[0] += weight * QQ(3/5) * QQ(Y)^2 * n3 * n2
        b[1] += weight * (QQ(1/2) * n3 if n2 == 2 else 0)
        b[2] += weight * (QQ(1/2) * n2 if n3 == 3 else 0)
    return b
gauge = [0, -QQ(22/3), -11]
f = contrib(fermions, QQ(2/3)); s = contrib(scalars, QQ(1/3))
b = [gauge[i] + 3 * f[i] + s[i] for i in range(3)]
print("per generation:", f, "  Higgs:", s)
print("b_i =", b)
