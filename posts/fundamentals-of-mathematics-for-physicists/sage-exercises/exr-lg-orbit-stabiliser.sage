# Exercise exr-lg-orbit-stabiliser: the rotations fixing the north pole e_3 of S^2: their generators X in so(3) with
# X e_3 = 0 span L_3, so the isotropy group is the SO(2) of rotations about Z; the SO(2)-orbit of a point is a parallel.
x1, x2, x3 = var("x1 x2 x3")
L = [matrix(SR, [[0, 0, 0], [0, 0, -1], [0, 1, 0]]), matrix(SR, [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]), matrix(SR, [[0, -1, 0], [1, 0, 0], [0, 0, 0]])]
X = x1 * L[0] + x2 * L[1] + x3 * L[2]
e3 = vector(SR, [0, 0, 1])
print("X e_3 =", X * e3, "; vanishes iff", solve(list(X * e3), [x1, x2]))
t, th = var("t th", domain="real")
R = (t * L[2]).exp().apply_map(lambda e: e.rectform().simplify_full())
print("R(t) e_3 = e_3:", R * e3 == e3)
p = vector(SR, [sin(th), 0, cos(th)])
orb = (R * p).simplify_full()
print("orbit of (sin th, 0, cos th) under SO(2):", orb, "; height constant:", bool(orb[2] == cos(th)), "; radius^2 =", (orb[0]^2 + orb[1]^2).simplify_full())
