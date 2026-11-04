# Exercise exr-lg-so2-group-law: the rotations R(t) about the Z-axis: R(0) = 1, R(s) R(t) = R(s + t), R R^T = 1,
# det R = 1 and R(t)^{-1} = R(-t), checked symbolically.
s, t = var("s t", domain="real")
def R(u): return matrix(SR, [[cos(u), -sin(u), 0], [sin(u), cos(u), 0], [0, 0, 1]])
print("R(0) = 1:", R(0) == identity_matrix(SR, 3))
D = (R(s) * R(t) - R(s + t)).apply_map(lambda e: e.trig_expand().simplify_full())
print("R(s) R(t) - R(s + t) =", D.list())
print("R R^T =", (R(t) * R(t).transpose()).simplify_full().list(), "; det R =", R(t).det().simplify_full())
print("R(t)^{-1} = R(-t) = R(t)^T:", bool(R(t).inverse().simplify_full() == R(-t)) and R(-t) == R(t).transpose())
print("R(2 pi) = 1:", R(2 * pi) == identity_matrix(SR, 3))
