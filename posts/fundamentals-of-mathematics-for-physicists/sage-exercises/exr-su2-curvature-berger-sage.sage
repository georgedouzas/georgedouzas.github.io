# Exercise exr-su2-curvature-berger-sage: the Berger spheres, the SU(2) metric with the Hopf fibre rescaled by lambda,
# g = (1/4)[dth^2 + sin^2 th dph^2 + lambda^2 (dps + cos th dph)^2]: R(lambda) = 8 - 2 lambda^2, Einstein only for lambda = 1.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
lam = var('lam', domain='real'); assume(lam > 0)
def comps(c):
    th = c[0]
    return [[QQ(1)/4, 0, 0], [0, (sin(th)**2 + lam**2*cos(th)**2)/4, lam**2*cos(th)/4], [0, lam**2*cos(th)/4, lam**2/4]]
M, X, g = chart_metric('th ph ps', comps, ranges='th:(0,pi) ph:(0,2*pi) ps:(0,4*pi)')
th, ph, ps = X[:]
print("lambda = 1 gives back the SU(2) metric:", all(bool((g[i, j].expr().subs(lam=1) - [[QQ(1)/4, 0, 0], [0, QQ(1)/4, cos(th)/4], [0, cos(th)/4, QQ(1)/4]][i][j]).simplify_full() == 0) for i in range(3) for j in range(3)))
Rs = g.ricci_scalar().expr().simplify_full()
print("scalar curvature R(lambda) =", Rs, ":", bool(Rs == 8 - 2*lam**2))
print("R(1) = 6:", bool(Rs.subs(lam=1) == 6), "; R -> 8 as lambda -> 0 (collapsing fibre), R < 0 for lambda > 2")
Ric = g.ricci()
ratio_th = (Ric[0, 0].expr()/g[0, 0].expr()).simplify_full(); ratio_ps = (Ric[2, 2].expr()/g[2, 2].expr()).simplify_full()
print("R_thth / g_thth =", ratio_th, ";  R_psps / g_psps =", ratio_ps)
print("Einstein (the two ratios agree) exactly when lambda = 1:", solve(ratio_th == ratio_ps, lam))
