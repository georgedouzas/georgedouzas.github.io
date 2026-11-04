# Exercise exr-sphere-conformal-fields: the gradient fields of the coordinate functions on the round sphere are
# conformal Killing fields, not Killing: for X = grad(z) = -sin(theta) d/dtheta, L_X g = -2 cos(theta) g = -2 z g,
# while the three rotation generators K_a have L_K g = 0.
M = Manifold(2, "S2", structure="Riemannian"); C = M.chart(r"th:(0,pi):\theta ph:(0,2*pi):\varphi"); th, ph = C[:]
g = M.metric("g"); g[0, 0] = 1; g[1, 1] = sin(th) ** 2
coords = {"x": sin(th) * cos(ph), "y": sin(th) * sin(ph), "z": cos(th)}
for name, F in coords.items():
    f = M.scalar_field(F, name=name)
    X = f.gradient(g)
    L = g.lie_derivative(X)
    print(f"grad {name} =", X.display(), f";  L_grad({name}) g = -2 {name} g:", L == -2 * f * g)
K = [M.vector_field(-sin(ph), -cos(th) / sin(th) * cos(ph)), M.vector_field(cos(ph), -cos(th) / sin(th) * sin(ph)), M.vector_field(0, 1)]
print("L_K_a g = 0 for a = 1, 2, 3:", [g.lie_derivative(k) == 0 for k in K])
