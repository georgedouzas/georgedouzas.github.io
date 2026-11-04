# Exercise exr-kkr-truncation-cubic: a cubic interaction lambda Phi^3 in five dimensions couples the modes n1, n2, n3
# through int_0^{2 pi R} e^{i (n1 + n2 + n3) y/R} dy = 2 pi R delta_{n1 + n2 + n3, 0}: momentum around the circle is
# conserved. Zero modes therefore source only the zero mode, while two massive modes n and -n source it.
y, R = var("y R"); assume(R > 0)
def overlap(a, b, c):
    return integrate(exp(I * (a + b + c) * y / R), y, 0, 2 * pi * R)
vals = {(a, b, c): overlap(a, b, c) for a in range(-2, 3) for b in range(-2, 3) for c in range(-2, 3)}
print("[PASS] the cubic overlap is 2 pi R if n1 + n2 + n3 = 0 and 0 otherwise (n_i in -2..2):",
      all(bool(v == (2 * pi * R if sum(k) == 0 else 0)) for k, v in vals.items()))
print("[PASS] (0, 0, n) couples only to n = 0, while (1, -1, 0) couples massive modes to the zero mode:",
      all(bool(vals[(0, 0, c)] == 0) for c in (-2, -1, 1, 2)) and bool(vals[(1, -1, 0)] == 2 * pi * R))
