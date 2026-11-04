# Exercise exr-gr-ds-killing: on de Sitter space ds^2 = dt^2 - e^{2Ht}(dx^2 + dy^2 + dz^2), signature (+, -, -, -),
# the vector field xi = d_t - H (x d_x + y d_y + z d_z) is a Killing vector: L_xi g = 0 (time translation combined with a
# dilation), while d_t alone is not.
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _curvature import *
H = var('H'); assume(H > 0)
M, X, g = chart_metric('t x y z', lambda c: [[1, 0, 0, 0], [0, -exp(2*H*c[0]), 0, 0], [0, 0, -exp(2*H*c[0]), 0], [0, 0, 0, -exp(2*H*c[0])]], structure='Lorentzian')
t, x, y, z = X[:]
xi = M.vector_field(1, -H*x, -H*y, -H*z, name='xi')
dt = M.vector_field(1, 0, 0, 0, name='dt')
print("L_xi g = 0:", g.lie_derivative(xi) == 0)
print("L_{d_t} g =", g.lie_derivative(dt).display())
