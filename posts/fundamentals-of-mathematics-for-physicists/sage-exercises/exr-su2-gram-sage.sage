# Exercise exr-su2-gram-sage: the Gram matrix of the Euler-angle parametrisation of SU(2) = S^3, from the Jacobian of
# (Re a, Im a, Re b, Im b) with a = cos(theta/2) e^{-i(phi+psi)/2}, b = -i sin(theta/2) e^{i(phi-psi)/2}:
# G = (1/4) [[1, 0, 0], [0, 1, cos theta], [0, cos theta, 1]] and det G = sin^2 theta / 64.
th, ph, ps = var("th ph ps", domain="real")
E = vector([cos(th/2)*cos((ph + ps)/2), -cos(th/2)*sin((ph + ps)/2), sin(th/2)*sin((ph - ps)/2), -sin(th/2)*cos((ph - ps)/2)])
print("lands on S^3:", bool((E*E).simplify_full() == 1))
J = jacobian(E, (th, ph, ps))
G = (J.transpose()*J).apply_map(lambda e: e.trig_reduce().simplify_full())
print("Gram matrix:"); print(G)
target = matrix(SR, [[1, 0, 0], [0, 1, cos(th)], [0, cos(th), 1]])/4
print("equals (1/4) [[1, 0, 0], [0, 1, cos theta], [0, cos theta, 1]]:", bool((G - target).simplify_full() == 0))
print("determinant:", G.det().simplify_full(), " equals sin^2 theta / 64:", bool((G.det() - sin(th)**2/64).simplify_full() == 0))
print("the angle between the phi- and psi-velocities has cosine G_{phi psi}/sqrt(G_{phi phi} G_{psi psi}) =", (G[1, 2]/sqrt(G[1, 1]*G[2, 2])).simplify_full())
