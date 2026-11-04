# Exercise exr-cr-exp-jacobian: the real Jacobian of (x, y) -> (e^x cos y, e^x sin y) is e^x times a rotation by y;
# its determinant is e^{2x} = |e^z|^2.
x, y = var("x y", domain="real")
u, v = exp(x)*cos(y), exp(x)*sin(y)
J = jacobian([u, v], [x, y])
print("J =\n", J)
print("det J =", J.det().simplify_full())
print("J^T J = e^(2x) * identity:", bool(((J.transpose()*J) - exp(2*x)*identity_matrix(2)).simplify_full() == 0))
R = matrix([[cos(y), -sin(y)], [sin(y), cos(y)]])
print("J == e^x * rotation(y):", bool((J - exp(x)*R).simplify_full() == 0))
