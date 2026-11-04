# Exercise exr-ssg-goldstino: at any stationary point of V = sum |W_i|^2 with F-terms only, dV/dphi_j = W^{ij}... precisely
# dV/dphi_j = sum_i W^{ij} conj(W^i) = 0, so the fermion mass matrix W^{ij} annihilates the vector conj(W^i) = -F_i: the
# goldstino direction G = F_i psi_i is massless. We check this for the O'Raifeartaigh model at the origin (m^2 > yk) and at
# the shifted minimum (m^2 < yk), with real parameters.
k, m, y = var("k m y", domain="positive")
p1, p2, p3 = var("p1 p2 p3")
W = -k * p1 + m * p2 * p3 + y / 2 * p1 * p3^2
ps = (p1, p2, p3)
Wi = vector(SR, [diff(W, p) for p in ps]); Wij = matrix(SR, 3, 3, lambda i, j: diff(W, ps[i], ps[j]))
def check(pt):
    Fv = -Wi.subs(pt); M = Wij.subs(pt)                 # real point: conjugation is trivial
    return Fv, M, (M * Fv).simplify_full()
Fv, M, MF = check({p1: 0, p2: 0, p3: 0})
print("origin: F =", Fv, " W^{ij} F_j =", MF)
print("[PASS] at the origin F = (k, 0, 0) and W^{ij} F_j = 0: the goldstino is psi_1:", Fv == vector([k, 0, 0]) and MF.is_zero())
assume(m^2 < y * k)
x3 = sqrt(2 * (y * k - m^2)) / y
x1 = var("x1"); x2 = -y * x1 * x3 / m                   # F_3 = 0 fixes phi_2 in terms of the flat phi_1
pt = {p1: x1, p2: x2, p3: x3}
Fv, M, MF = check(pt)
print("shifted minimum: F =", [f.simplify_full() for f in Fv], " W^{ij} F_j =", MF)
print("[PASS] at the shifted minimum (m^2 < yk) W^{ij} F_j = 0 as well: a massless goldstino along F:", MF.is_zero())
print("[PASS] the F-vector there is non-zero, (m^2/y, -m x3, 0) up to sign: supersymmetry is still broken:",
      bool((Fv[0] - m^2 / y).simplify_full() == 0) and bool((Fv[1] + m * x3).simplify_full() == 0) and bool(Fv[2].simplify_full() == 0))
