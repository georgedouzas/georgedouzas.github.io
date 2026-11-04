# Exercise exr-cs-isotropy-conjugate: the isotropy groups of SO(3) acting on the sphere are conjugate. For the
# rotation g = exp(phi L_3) exp(theta L_1) sending the pole o to p = g . o, the isotropy algebra at p is spanned by
# g L_3 g^{-1}, which annihilates p and equals p_1 L_1 + p_2 L_2 + p_3 L_3, the generator of rotations about p.
th, ph, s = var("th ph s", domain="real")
L = [matrix(SR, [[0,0,0],[0,0,-1],[0,1,0]]), matrix(SR, [[0,0,1],[0,0,0],[-1,0,0]]), matrix(SR, [[0,-1,0],[1,0,0],[0,0,0]])]
simp = lambda M: M.apply_map(lambda e: e.rectform().simplify_full())
o = vector(SR, [0, 0, 1])
g = simp((ph * L[2]).exp() * (th * L[0]).exp())
p = (g * o).apply_map(lambda e: e.simplify_full())
K = simp(g * L[2] * g.inverse())
print("p = g . o =", p.list())
print("g L_3 g^{-1} =", K.rows())
print("(g L_3 g^{-1}) p = 0, so exp(s g L_3 g^{-1}) = g exp(s L_3) g^{-1} fixes p:", bool((K * p).apply_map(lambda e: e.simplify_full()) == 0))
print("g L_3 g^{-1} = p_1 L_1 + p_2 L_2 + p_3 L_3 (rotations about the axis through p):", bool(simp(K - sum(p[a] * L[a] for a in range(3))) == 0))
print("the stabiliser of p is g SO(2) g^{-1}: conjugate to the stabiliser of the pole")
