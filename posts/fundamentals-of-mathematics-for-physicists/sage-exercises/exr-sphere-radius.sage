# Exercise exr-sphere-radius: curvature of the sphere of radius R (SageManifolds).
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
R = var('R', domain='real'); assume(R > 0)
M, X, g = chart_metric('th ph', lambda c: [[R**2, 0], [0, R**2*sin(c[0])**2]], ranges='th:(0,pi) ph:(0,2*pi)')
Rm, Ric, Rs = curvature(g)
print("R^theta_{phi theta phi} =", Rm[0, 1, 0, 1].expr(), "   R^phi_{theta theta phi} =", Rm[1, 0, 0, 1].expr())
print("Ricci tensor:", Ric.display())
print("Ricci = g / R^2:", (Ric - g/R**2) == 0)
print("scalar curvature:", Rs.expr().simplify_full())
