# Exercise exr-fbun-so3-frames: SO(3) -> S^2, R -> R e_3, is the orthonormal frame bundle of the sphere. In Euler
# angles R = R_z(phi) R_y(theta) R_z(psi): the third column is the point of S^2, the first two columns are an
# orthonormal basis of its tangent plane, and the right action of R_z(alpha) rotates them in that plane by alpha.
th, ph, ps, al = var("th ph ps al", domain="real")
def Rz(a): return matrix([[cos(a), -sin(a), 0], [sin(a), cos(a), 0], [0, 0, 1]])
def Ry(a): return matrix([[cos(a), 0, sin(a)], [0, 1, 0], [-sin(a), 0, cos(a)]])
R = Rz(ph) * Ry(th) * Rz(ps)
n = R.column(2)
print("R e_3 =", n)
print("R is orthogonal with det 1:", bool((R.T * R - identity_matrix(3)).simplify_full() == 0), bool(R.det().simplify_full() == 1))
e1, e2 = R.column(0), R.column(1)
print("e1, e2 are tangent to S^2 at R e_3:", bool(e1.dot_product(n).simplify_full() == 0), bool(e2.dot_product(n).simplify_full() == 0))
Ra = R * Rz(al)
print("the right action of R_z(alpha) fixes the base point:", bool((Ra.column(2) - n).simplify_full() == 0))
print("and rotates the frame: e1' = cos(al) e1 + sin(al) e2:",
      bool((Ra.column(0) - cos(al) * e1 - sin(al) * e2).simplify_full() == 0))
print("at psi = 0 the frame is (d/dtheta, d/dphi / sin theta):", R.column(0).subs(ps=0), R.column(1).subs(ps=0))
