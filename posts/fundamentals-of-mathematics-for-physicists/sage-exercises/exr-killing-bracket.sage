# Exercise exr-killing-bracket: the brackets of the three Killing fields of the sphere close into so(3) with the
# sign of an anti-representation, [K_a, K_b] = -eps_abc K_c (SageManifolds vector fields and their bracket).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('th ph', lambda c: [[1, 0], [0, sin(c[0])**2]], ranges='th:(0,pi) ph:(0,2*pi)')
th, ph = X[:]
K = [M.vector_field(-sin(ph), -cos(th)/sin(th)*cos(ph), name='K1'), M.vector_field(cos(ph), -cos(th)/sin(th)*sin(ph), name='K2'),
     M.vector_field(0, 1, name='K3')]
for a, b, c in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    br = K[a].bracket(K[b])
    print(f"[K_{a+1}, K_{b+1}] =", br.display(), f"= -K_{c+1}:", (br + K[c]) == 0)
