# Exercise exr-gr-variation-minisuperspace: the Einstein-Hilbert action on ds^2 = N(t)^2 dt^2 - a(t)^2 dx^2 (signature
# (+, -, -, -)) plus the dust action -int sqrt(-g) rho with rho a^3 = rho0 constant. Varying the lapse N and setting
# N = 1 gives the first Friedmann equation adot^2/a^2 = (kappa^2/3) rho.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M = Manifold(4, 'M', structure='Lorentzian'); X = M.chart('t x y z'); t = X[0]
a = function('a')(t); N = function('N')(t); kappa, rho0 = var('kappa rho0')
g = M.metric('g'); g[0, 0] = N^2
for i in (1, 2, 3): g[i, i] = -a^2
sqrtg = N*a^3
R = g.ricci_scalar().expr()
L = -sqrtg*R/(2*kappa^2) - N*rho0                          # dust: sqrt(-g) rho = N a^3 rho = N rho0
# Euler-Lagrange derivative with respect to N (L contains N, N', a, a', a'')
n0, n1, a0, a1, a2 = var('n0 n1 a0 a1 a2')
rep = {diff(a, t, 2): a2, diff(a, t): a1, diff(N, t): n1}
Ls = L.subs(rep).subs({a: a0, N: n0})
back = {n0: N, n1: diff(N, t), a0: a, a1: diff(a, t), a2: diff(a, t, 2)}
EL_N = diff(Ls, n0).subs(back) - diff(diff(Ls, n1).subs(back), t)
res = EL_N.subs({diff(N, t): 0}).subs({N: 1}).simplify_full()
print("dL/dN at N = 1:", res)
print("vanishes exactly when adot^2/a^2 = kappa^2 rho/3 with rho = rho0/a^3:", bool((res - (3*diff(a, t)^2*a/kappa^2 - rho0)).simplify_full() == 0))
