# Exercise exr-lg-su2-dual-frame: the left-invariant vector fields xi_a of SU(2) in Euler angles, obtained by inverting
# the matrix of components of the forms sigma^a, and their brackets [xi_a, xi_b] = eps_abc xi_c.
G = Manifold(3, "SU2"); X.<th, ph, ps> = G.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi ps:(0,4*pi):\psi")
S = matrix(SR, [[cos(ps), sin(th) * sin(ps), 0], [-sin(ps), sin(th) * cos(ps), 0], [0, cos(th), 1]])
Si = S.inverse().simplify_full()
xi = [G.vector_field(*[Si[i, a] for i in range(3)], name=f"xi{a+1}") for a in range(3)]
for a in range(3):
    print(xi[a].display())
sig = [G.one_form(*S.row(a)) for a in range(3)]
print("sigma^a(xi_b) = delta:", matrix(SR, 3, 3, lambda a, b: sig[a](xi[b]).expr().simplify_full()) == identity_matrix(SR, 3))
for (a, b, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    print(f"[xi{a+1}, xi{b+1}] = xi{c+1}:", xi[a].bracket(xi[b]) == xi[c])
