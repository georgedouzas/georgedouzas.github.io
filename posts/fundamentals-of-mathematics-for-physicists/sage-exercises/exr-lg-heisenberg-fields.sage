# Exercise exr-lg-heisenberg-fields: the left-invariant vector fields of the Heisenberg group h(x, y, z): the velocity
# of t -> g e^{tA} for A = E_12, E_23, E_13 gives X = d_x, Y = d_y + x d_z, Z = d_z, with [X, Y] = Z as [E_12, E_23] = E_13;
# they are invariant under the left translations (x, y, z) -> (x + a, y + b, z + c + a y).
x, y, z, t, a, b, c = var("x y z t a b c", domain="real")
def h(u, v, w): return matrix(SR, [[1, u, w], [0, 1, v], [0, 0, 1]])
E12, E23, E13 = [matrix(SR, 3, 3, {(i, j): 1}) for (i, j) in [(0, 1), (1, 2), (0, 2)]]
for name, A in [("E12", E12), ("E23", E23), ("E13", E13)]:
    V = diff(h(x, y, z) * (t * A).exp(), t).subs(t=0)
    print(f"velocity of g exp(t {name}) in (x, y, z):", (V[0, 1], V[1, 2], V[0, 2]))
print("[E12, E23] = E13:", E12 * E23 - E23 * E12 == E13)
M = Manifold(3, "H", structure="smooth"); C.<x, y, z> = M.chart()
X = M.vector_field(1, 0, 0, name="X"); Y = M.vector_field(0, 1, x, name="Y"); Z = M.vector_field(0, 0, 1, name="Z")
print("[X, Y] =", X.bracket(Y).display(), "; [X, Z] =", X.bracket(Z).display(), "; [Y, Z] =", Y.bracket(Z).display())
var("a b c")
Lh = M.diff_map(M, [x + a, y + b, z + c + a * y], name="L_h")
J = Lh.jacobian_matrix()
print("Jacobian of L_h:", J[:])
for name, comps in [("X", (1, 0, 0)), ("Y", (0, 1, x)), ("Z", (0, 0, 1))]:
    pushed = J[:] * vector(SR, [SR(e) for e in comps])
    target = vector(SR, comps).subs(x=x + a, y=y + b, z=z + c + a * y)
    print(f"dL_h({name}) at g = {name} at hg:", bool(pushed == target))
