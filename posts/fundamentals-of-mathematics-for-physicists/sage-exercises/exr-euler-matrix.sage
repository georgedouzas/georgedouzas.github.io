# Exercise exr-euler-matrix: the Euler-angle element g = exp(phi T_3) exp(theta T_1) exp(psi T_3) of SU(2) with Sage's
# matrix exponential: its entries a, b, the condition |a|^2 + |b|^2 = 1, and the degeneration at theta = 0.
th, ph, ps = var("th ph ps", domain="real")
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
T = [-I / 2 * sa for sa in s]
g = ((ph * T[2]).exp() * (th * T[0]).exp() * (ps * T[2]).exp()).apply_map(lambda e: e.exponentialize().simplify_full())
zero = lambda e: bool(e.exponentialize().simplify_full() == 0)
print("g(theta, phi, psi) ="); print(g)
a, b = g[0, 0], g[1, 0]
print("a = cos(theta/2) e^{-i(phi+psi)/2}:", zero(a - cos(th/2) * exp(-I*(ph + ps)/2)))
print("b = -i sin(theta/2) e^{i(phi-psi)/2}:", zero(b + I * sin(th/2) * exp(I*(ph - ps)/2)))
print("|a|^2 + |b|^2 = 1:", zero(a * a.conjugate() + b * b.conjugate() - 1), "; det g = 1:", zero(g.det() - 1))
print("at theta = 0, g depends only on phi + psi:", all(zero(e) for e in (g.subs(th=0) - ((ph + ps) * T[2]).exp()).list()))
