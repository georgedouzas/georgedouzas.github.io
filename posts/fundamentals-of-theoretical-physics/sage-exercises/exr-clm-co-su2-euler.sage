# Exercise exr-clm-co-su2-euler: tr U/2 = cos(theta/2) cos((phi + psi)/2) for U = exp(phi T3) exp(theta T1) exp(psi T3),
# T_a = -i sigma_a/2.
th, ph, ps = var("theta phi psi", domain="real")
s1 = matrix(SR, [[0, 1], [1, 0]]); s3 = matrix(SR, [[1, 0], [0, -1]])
T1, T3 = -I / 2 * s1, -I / 2 * s3
U = (ph * T3).exp() * (th * T1).exp() * (ps * T3).exp()
half_tr = (U.trace() / 2).simplify_full()
print("tr U / 2 =", half_tr)
print("== cos(theta/2) cos((phi + psi)/2):", bool((half_tr - (cos(th / 2) * cos((ph + ps) / 2)).exponentialize()).simplify_full() == 0))
print("U at theta = 0, psi = -phi:", U.subs(theta=0, psi=-ph).simplify_full())
