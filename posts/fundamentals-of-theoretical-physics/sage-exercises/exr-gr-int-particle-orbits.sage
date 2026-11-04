# Exercise exr-gr-int-particle-orbits: d_t and d_phi are Killing vectors of the Schwarzschild metric (signature
# (+, -, -, -), G = 1), so E = g(d_t, u) = (1 - 2m/r) t' and L = -g(d_phi, u) = r^2 sin^2(th) phi' are conserved along
# geodesics; at large r and low speed the radial equation becomes the Newtonian energy equation with
# U_eff = -m/r + L^2/(2 r^2), plus the relativistic -m L^2/r^3.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
m = var('m'); assume(m > 0)
M, X, g = chart_metric('t r th ph', lambda c: [[1 - 2*m/c[1], 0, 0, 0], [0, -1/(1 - 2*m/c[1]), 0, 0], [0, 0, -c[1]^2, 0], [0, 0, 0, -c[1]^2*sin(c[2])^2]], structure='Lorentzian', ranges='t r:(0,+oo) th:(0,pi) ph:(0,2*pi)')
dt = M.vector_field(1, 0, 0, 0); dph = M.vector_field(0, 0, 0, 1)
print("L_{d_t} g = 0:", g.lie_derivative(dt) == 0, "; L_{d_phi} g = 0:", g.lie_derivative(dph) == 0)
r, L, E = var('r L E')
V = (1 - 2*m/r)*(1 + L^2/r^2)
print("(V - 1)/2 =", ((V - 1)/2).expand(), " = -m/r + L^2/(2r^2) - m L^2/r^3")
