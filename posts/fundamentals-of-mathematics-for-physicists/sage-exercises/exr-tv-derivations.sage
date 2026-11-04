# Exercise exr-tv-derivations: a vector field with arbitrary components obeys the Leibniz rule on arbitrary functions,
# annihilates constants and is determined by its values on the coordinate functions; a second derivative is not a
# tangent vector, its Leibniz defect being the cross term 2 f_theta g_theta.
S2 = Manifold(2, "S^2"); A = S2.open_subset("A")
sph = A.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = sph[:]
a, b = function("a")(th, ph), function("b")(th, ph)
X = A.vector_field(a, b, name="X")
f = A.scalar_field(function("f")(th, ph), name="f"); g = A.scalar_field(function("g")(th, ph), name="g")
print("Leibniz rule X(fg) = X(f) g + f X(g):", bool((X(f*g) - X(f)*g - f*X(g)).expr().simplify_full() == 0))
print("X annihilates the constant function 1:", bool(X(A.scalar_field(1)).expr() == 0))
print("X(theta) = a and X(phi) = b, the components:", bool(X(A.scalar_field(th)).expr() == a), bool(X(A.scalar_field(ph)).expr() == b))
D = lambda h: diff(h, th, 2)
F, G = f.expr(), g.expr()
print("Leibniz defect of the second derivative d^2/dtheta^2:", (D(F*G) - D(F)*G - F*D(G)).simplify_full())
