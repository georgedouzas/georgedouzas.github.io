# Exercise exr-sr-velocity-space: the metric induced by minus the Minkowski form on the unit hyperboloid of
# four-velocities is the metric of three-dimensional hyperbolic space.
var('chi th ph', domain='real')
X = vector([cosh(chi), sinh(chi)*sin(th)*cos(ph), sinh(chi)*sin(th)*sin(ph), sinh(chi)*cos(th)])
eta = diagonal_matrix([1, -1, -1, -1])
J = matrix([[diff(X[m], q) for q in (chi, th, ph)] for m in range(4)])
h = (-J.transpose()*eta*J).simplify_full()
print("induced metric (chi, theta, phi):"); print(h)
print("on the hyperboloid:", (X*eta*X).simplify_full())
