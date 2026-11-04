# Exercise exr-einstein-tensor-su2-sage: the Einstein tensor of the SU(2) metric is G = -g, its trace is -3 = (1 - n/2) R,
# and its covariant divergence vanishes (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('th ph ps', lambda c: [[QQ(1)/4, 0, 0], [0, QQ(1)/4, cos(c[0])/4], [0, cos(c[0])/4, QQ(1)/4]], ranges='th:(0,pi) ph:(0,2*pi) ps:(0,4*pi)')
Ric = g.ricci(); Rs = g.ricci_scalar()
G = Ric - Rs*g/2; G.set_name('G')
print("Einstein tensor:", G.display())
print("G = -g:", G == -g)
trG = sum(g.inverse()[i, j].expr()*G[i, j].expr() for i in range(3) for j in range(3)).simplify_full()
print("trace g^{ij} G_ij =", trG, "= (1 - n/2) R with n = 3, R = 6:", bool(trG == (1 - QQ(3)/2)*6))
nab = g.connection()
divG = nab(G.up(g, 0))['^a_ba']                                # nabla_a G^a_b
print("divergence nabla^i G_ij:", divG.display(), "-> vanishes:", divG == 0)
