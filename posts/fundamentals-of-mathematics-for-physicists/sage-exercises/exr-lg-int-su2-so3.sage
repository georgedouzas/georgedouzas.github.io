# Exercise exr-lg-int-su2-so3: from SU(2) to SO(3) end to end. For the Euler-angle element g of SU(2), the matrix of
# Ad_g in the basis T_a is the Euler-angle rotation e^{phi L_3} e^{theta L_1} e^{psi L_3}; it is orthogonal of
# determinant one, Ad_{-g} = Ad_g, its derivative ad_{T_a} is L_a, and the Killing form is -2 delta.
th, ph, ps = var("th ph ps", domain="real")
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
T = [-I / 2 * sa for sa in s]
L = [matrix(SR, [[0, 0, 0], [0, 0, -1], [0, 1, 0]]), matrix(SR, [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]), matrix(SR, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])]
def expT(a, u): return cos(u/2) * identity_matrix(SR, 2) - I * sin(u/2) * s[a]
def rot(a, u): return identity_matrix(SR, 3) + sin(u) * L[a] + (1 - cos(u)) * L[a]^2
def simp(e): return e.exponentialize().simplify_full().trig_reduce().simplify_full()
g = expT(2, ph) * expT(0, th) * expT(2, ps); gi = g.conjugate_transpose()
Ad = matrix(SR, 3, 3, lambda a, b: simp(-2 * (T[a] * g * T[b] * gi).trace()))
R = rot(2, ph) * rot(0, th) * rot(2, ps)
print("Ad_g = e^{phi L_3} e^{theta L_1} e^{psi L_3}:", bool((Ad - R).apply_map(lambda e: e.trig_expand().simplify_full()) == 0))
print("Ad_g^T Ad_g = 1:", bool((Ad.transpose() * Ad).apply_map(lambda e: e.trig_expand().simplify_full()) == 1), "; det Ad_g =", Ad.det().trig_expand().simplify_full())
Adm = matrix(SR, 3, 3, lambda a, b: simp(-2 * (T[a] * (-g) * T[b] * (-gi)).trace()))
print("Ad_{-g} = Ad_g:", bool(Adm == Ad))
ad = [matrix(SR, 3, 3, lambda c, b: -2 * (T[c] * (T[a] * T[b] - T[b] * T[a])).trace()) for a in range(3)]
print("ad_{T_a} = L_a:", all(ad[a] == L[a] for a in range(3)), "; Killing form:", matrix(SR, 3, 3, lambda a, b: (ad[a] * ad[b]).trace()).list())
print("at (theta, phi, psi) = (0, 0, 2 pi): g =", g.subs(th=0, ph=0, ps=2 * pi).simplify_full().list(), ", Ad_g =", Ad.subs(th=0, ph=0, ps=2 * pi).list())
