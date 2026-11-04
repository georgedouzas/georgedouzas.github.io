# Exercise exr-killing-length: the length of each Killing field vanishes exactly on its rotation axis (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
M, X, g = chart_metric('th ph', lambda c: [[1, 0], [0, sin(c[0])**2]], ranges='th:(0,pi) ph:(0,2*pi)')
th, ph = X[:]
K = [M.vector_field(-sin(ph), -cos(th)/sin(th)*cos(ph)), M.vector_field(cos(ph), -cos(th)/sin(th)*sin(ph)), M.vector_field(0, 1)]
for a in range(3):
    print(f"|K_{a+1}|^2 = g(K, K) =", g(K[a], K[a]).expr().simplify_full().trig_expand().simplify_full())
