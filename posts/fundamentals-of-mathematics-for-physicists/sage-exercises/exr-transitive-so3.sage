# Exercise exr-transitive-so3: the action of SO(3) on the sphere is transitive and the isotropy group of the north
# pole is SO(2): with Sage's matrix exponential, the rotation exp(phi L_3) exp(theta L_1) sends the pole to the point
# with spherical coordinates (theta, phi), and exp(t L_3) fixes the pole.
th, ph, t = var("th ph t", domain="real")
L1 = matrix(SR, [[0, 0, 0], [0, 0, -1], [0, 1, 0]]); L3 = matrix(SR, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])
o = vector(SR, [0, 0, 1])
R3 = (t * L3).exp().apply_map(lambda e: e.rectform().simplify_full())
print("exp(t L_3) o =", (R3 * o).list(), " -> the pole is fixed: the isotropy group contains all rotations about Z")
g = ((ph * L3).exp() * (th * L1).exp()).apply_map(lambda e: e.rectform().simplify_full())
p = (g * o).apply_map(lambda e: e.simplify_full())
print("exp(phi L_3) exp(theta L_1) o =", p.list())
print("= (sin th sin ph?, ...) matches a point of the sphere with colatitude theta:", bool((p[2] - cos(th)).simplify_full() == 0) and bool((p[0]**2 + p[1]**2 - sin(th)**2).simplify_full() == 0))
print("every point of the sphere is reached (transitive): choose theta, phi as its spherical coordinates")
print("the coset of g is {g h : h = exp(t L_3)}: all these send o to the same point:", bool(((g * R3 * o) - p).apply_map(lambda e: e.simplify_full()) == 0))
