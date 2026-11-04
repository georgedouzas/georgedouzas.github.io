# Exercise exr-fbun-normal-frame: TS^2 plus the normal line bundle is the trivial bundle S^2 x R^3: at every point
# of the sphere (away from the poles) the unit vectors e_theta, e_phi and the normal n form an orthonormal
# positively oriented basis of R^3, and the constant sections e_x, e_y, e_z of R^3 decompose along it.
th, ph = var("th ph", domain="real")
n = vector([sin(th) * cos(ph), sin(th) * sin(ph), cos(th)])
eth = diff(n, th)
eph = (diff(n, ph) / sin(th)).simplify_full()
M = matrix([eth, eph, n]).simplify_full()
print("rows e_theta, e_phi, n:"); print(M)
print("orthonormal:", bool((M * M.T).simplify_full() == identity_matrix(3)))
print("determinant:", M.det().simplify_full())
ez = vector([0, 0, 1])
print("e_z = (e_z . e_theta) e_theta + (e_z . e_phi) e_phi + (e_z . n) n with coefficients",
      [ez.dot_product(r).simplify_full() for r in (eth, eph, n)])
print("tangential part of e_z vanishes only at the poles: -sin(theta) = 0 at theta =", solve(-sin(th) == 0, th))
