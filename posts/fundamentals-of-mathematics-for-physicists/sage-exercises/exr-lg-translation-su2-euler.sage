# Exercise exr-lg-translation-su2-euler: left and right translation by e^{tT_3} on the Euler-angle element
# g(theta, phi, psi) = e^{phi T_3} e^{theta T_1} e^{psi T_3} shift phi and psi respectively, while left translation by
# e^{tT_1} is not a shift of any single angle.
th, ph, ps, t = var("th ph ps t", domain="real")
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
def expT(a, u): return cos(u/2) * identity_matrix(SR, 2) - I * sin(u/2) * s[a]
def g(a, b, c): return expT(2, b) * expT(0, a) * expT(2, c)
def same(A, B): return bool((A - B).apply_map(lambda e: e.exponentialize().simplify_full()) == 0)
print("e^{tT_3} g(th, ph, ps) = g(th, ph + t, ps):", same(expT(2, t) * g(th, ph, ps), g(th, ph + t, ps)))
print("g(th, ph, ps) e^{tT_3} = g(th, ph, ps + t):", same(g(th, ph, ps) * expT(2, t), g(th, ph, ps + t)))
print("e^{tT_3} g = g e^{tT_3}? (left = right?):", same(expT(2, t) * g(th, ph, ps), g(th, ph, ps) * expT(2, t)))
print("e^{tT_1} g(0, 0, 0) = g(t, 0, 0):", same(expT(0, t) * g(0, 0, 0), g(t, 0, 0)), "; but e^{tT_1} g(th, ph, 0) at ph = pi/2 equals g(th + t, pi/2, 0)?", same(expT(0, t) * g(th, pi/2, 0), g(th + t, pi/2, 0)))
