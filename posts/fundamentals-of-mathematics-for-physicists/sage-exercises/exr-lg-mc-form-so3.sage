# Exercise exr-lg-mc-form-so3: the Maurer-Cartan form R^{-1} dR of the Euler-angle rotation
# R = e^{phi L_3} e^{theta L_1} e^{psi L_3} in SO(3), with components sigma^a = -(1/2) tr(L_a R^{-1} dR), reproduces the
# left-invariant forms of SU(2): the double cover identifies the two Maurer-Cartan forms.
th, ph, ps = var("th ph ps", domain="real")
L = [matrix(SR, [[0, 0, 0], [0, 0, -1], [0, 1, 0]]), matrix(SR, [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]), matrix(SR, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])]
def rot(a, u): return identity_matrix(SR, 3) + sin(u) * L[a] + (1 - cos(u)) * L[a]^2
R = rot(2, ph) * rot(0, th) * rot(2, ps)
print("tr(L_a L_b) = -2 delta_ab:", matrix(SR, 3, 3, lambda a, b: (L[a] * L[b]).trace()) == -2 * identity_matrix(SR, 3))
coords = [th, ph, ps]
sig = matrix(SR, 3, 3, lambda a, i: (-(L[a] * R.transpose() * diff(R, coords[i])).trace() / 2).simplify_full().trig_reduce().simplify_full())
print("components sigma^a_i (rows a, columns theta, phi, psi):"); print(sig)
expected = matrix(SR, [[cos(ps), sin(th) * sin(ps), 0], [-sin(ps), sin(th) * cos(ps), 0], [0, cos(th), 1]])
print("equal to the SU(2) forms:", bool((sig - expected).apply_map(lambda e: e.trig_expand().simplify_full()) == 0))
D = [(R.transpose() * diff(R, c)).simplify_full() for c in coords]
print("R^{-1} dR - sigma^a L_a = 0 in each direction:", all(bool((D[i] - sum(expected[a, i] * L[a] for a in range(3))).apply_map(lambda e: e.trig_expand().simplify_full()) == 0) for i in range(3)))
