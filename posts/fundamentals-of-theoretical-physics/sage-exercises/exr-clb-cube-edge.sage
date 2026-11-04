# Exercise exr-clb-cube-edge (Sinha, Sec. 8.2.18): a uniform cube of side 2a rotating with angular velocity omega about a
# body diagonal; the diagonal is released and an edge that does not meet it is suddenly fixed. The angular momentum about
# the edge is conserved through the impulse; inertia tensors by integration.
x, y, z, a, M, w = var('x y z a M w'); assume(a > 0)
rho = M / (2 * a)^3
def I_tensor(c):   # inertia tensor about the point c, cube [-a, a]^3
    r = vector([x - c[0], y - c[1], z - c[2]])
    T = matrix(SR, 3, 3, lambda i, j: (r * r if i == j else 0) - r[i] * r[j])
    return T.apply_map(lambda e: integrate(integrate(integrate(rho * e, x, -a, a), y, -a, a), z, -a, a))
IG = I_tensor((0, 0, 0))
print("inertia tensor at the centre:", IG, " (isotropic, 2 M a^2/3)")
d = vector([1, 1, 1]) / sqrt(3)          # the diagonal through (-a,-a,-a) and (a,a,a)
L = IG * (w * d)                          # angular momentum about the centre (the centre is at rest)
e = vector([1, 0, 0])                     # the edge y = a, z = -a along x does not meet this diagonal
I_edge = (I_tensor((0, a, -a)) * e) * e
Le = L * e                                # L about any point of the edge = L about G, since G is at rest
w2 = (Le / I_edge).simplify_full()
print("edge y = a, z = -a meets the diagonal:", bool(solve([y == a, z == -a, x == y, y == z], [x, y, z]) != []))
print("moment of inertia about the edge:", I_edge, "  new angular velocity:", w2, " = sqrt(3)/12 omega:", bool((w2 - sqrt(3) / 12 * w).simplify_full() == 0))
print("kinetic energy ratio after/before:", ((I_edge * w2^2) / (L * (w * d))).simplify_full())
