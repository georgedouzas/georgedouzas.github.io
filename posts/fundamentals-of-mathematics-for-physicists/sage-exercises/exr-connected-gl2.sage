# Exercise exr-connected-gl2: the sign of the determinant separates GL(2, R); a path of rotations in SO(2).
t = var("t")
R = matrix(SR, [[cos(t), -sin(t)], [sin(t), cos(t)]]); S = matrix(SR, [[1, 0], [0, -1]])
print("det R(t) =", R.det().simplify_trig(), "; det of a reflection:", S.det())
print("[PASS] rotations have det +1 along the whole path, reflections det -1: two components of O(2):", bool(R.det().simplify_trig() == 1) and bool(S.det() == -1))
