# Exercise exr-inverse-rotation-matrix: R(alpha)^{-1} = R(-alpha) = R(alpha)^T for a rotation about the Z-axis.
a = var("a")
R = lambda t: matrix(SR, [[cos(t), -sin(t), 0], [sin(t), cos(t), 0], [0, 0, 1]])
print("R(a) R(-a) =", (R(a) * R(-a)).simplify_full())
print("[PASS] R(a)^{-1} = R(-a) = R(a)^T:", (R(a) * R(-a)).simplify_full() == identity_matrix(3) and R(-a) == R(a).transpose())
