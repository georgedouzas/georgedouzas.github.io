# Exercise exr-double-cover: the adjoint action of SU(2) on su(2) gives rotations, and g and -g give the same one.
t = var('t', domain='real')
s = [matrix(SR, [[0,1],[1,0]]), matrix(SR, [[0,-I],[I,0]]), matrix(SR, [[1,0],[0,-1]])]
T = [-I/2*sa for sa in s]
g = cos(t/2)*identity_matrix(SR, 2) - I*sin(t/2)*s[2]          # exp(t T_3)
# Ad_g T_b = g T_b g^{-1} = sum_a R_ab T_a ; read off R_ab = -2 Tr(T_a g T_b g^{-1})
R = matrix(SR, 3, 3, lambda a, b: (-2*(T[a]*g*T[b]*g.conjugate_transpose()).trace()).simplify_full().trig_reduce())
print("Ad(exp(t T3)) in the basis T_a ="); print(R)
print("rotation by t about the third axis:", bool(R == matrix(SR, [[cos(t), -sin(t), 0], [sin(t), cos(t), 0], [0, 0, 1]])))
print("exp(2 pi T3) = -1:", bool(g.subs(t=2*pi) == -identity_matrix(SR, 2)), "  Ad(exp(2 pi T3)) = 1:", bool(R.subs(t=2*pi) == identity_matrix(SR, 3)))
