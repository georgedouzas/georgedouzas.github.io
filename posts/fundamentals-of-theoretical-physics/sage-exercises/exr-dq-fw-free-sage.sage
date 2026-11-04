# Exercise exr-dq-fw-free-sage: U = cos(theta) + beta alpha_3 sin(theta) diagonalises H = alpha_3 p + beta m when tan(2 theta) = p/m;
# for p = 3, m = 4 the result is 5 beta.
one = identity_matrix(SR, 2); Z = zero_matrix(SR, 2); s3 = matrix(SR, [[1, 0], [0, -1]])
beta = block_matrix([[one, Z], [Z, -one]], subdivide=False); a3 = block_matrix([[Z, s3], [s3, Z]], subdivide=False)
p, m, th = var("p m theta", domain="real")
U = cos(th) * identity_matrix(SR, 4) + sin(th) * beta * a3
H = p * a3 + m * beta
Hp = (U * H * U.conjugate_transpose()).simplify_full().simplify_trig()
odd = ((Hp - beta * Hp * beta) / 2).simplify_full()
print("U H U^dag =", Hp.list())
print("odd part = alpha_3 (p cos 2theta - m sin 2theta):", (odd - (p * cos(2 * th) - m * sin(2 * th)) * a3).simplify_full().simplify_trig() == 0)
# p = 3, m = 4: cos 2theta = 4/5, sin 2theta = 3/5, so cos theta = 3/sqrt 10, sin theta = 1/sqrt 10
U0 = 3 / sqrt(10) * identity_matrix(SR, 4) + 1 / sqrt(10) * beta * a3
print("for p = 3, m = 4: U H U^dag =", (U0 * H.subs(p=3, m=4) * U0.conjugate_transpose()).simplify_full().diagonal(), "= 5 beta")
