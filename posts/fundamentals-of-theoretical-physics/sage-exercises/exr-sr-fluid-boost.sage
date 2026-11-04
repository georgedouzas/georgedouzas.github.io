# Exercise exr-sr-fluid-boost (Blennow-Ohlsson, Problems 1.147-1.148): the stress tensor of a perfect fluid seen from a
# frame moving with velocity v along x.
var('rho p v', domain='positive')
g = 1/sqrt(1 - v^2)
Lam = matrix(SR, [[g, -g*v, 0, 0], [-g*v, g, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
T = diagonal_matrix(SR, [rho, p, p, p])
Tp = (Lam*T*Lam.transpose()).simplify_full()
print("T'^00 =", Tp[0, 0], "   T'^01 =", Tp[0, 1], "   T'^11 =", Tp[1, 1])
eta = diagonal_matrix([1, -1, -1, -1])
print("trace before and after:", (eta*T).trace(), "|", (eta*Tp).trace().simplify_full())
print("radiation p = rho/3: T'^00 =", Tp[0, 0].subs(p=rho/3).factor())
