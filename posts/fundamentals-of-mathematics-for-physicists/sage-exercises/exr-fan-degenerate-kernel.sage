# Exercise exr-fan-degenerate-kernel: u(x) - lam * int_0^1 (1 + x y) u(y) dy = f(x), with u = f + lam (a + b x).
lam, a, b, x, y = var("lam a b x y")
f = SR(1)
u = f + lam * (a + b * x)
eqs = [a == integrate(u, x, 0, 1), b == integrate(x * u, x, 0, 1)]
M = matrix(SR, [[1 - lam, -lam / 2], [-lam / 2, 1 - lam / 3]])
d = M.det().expand()
print("determinant:", d, " = ", (12 * d).expand(), "/ 12")
print("singular values of lambda:", solve(d == 0, lam))
sol = solve(eqs, a, b, solution_dict=True)[0]
print("a =", sol[a].factor(), "  b =", sol[b].factor())
uf = u.subs(sol).factor()
print("u(x) =", uf)
print("check of the integral equation:", bool((uf - lam * integrate((1 + x * y) * uf.subs(x=y), y, 0, 1) - f).simplify_full() == 0))
print("reciprocal eigenvalues of the kernel matrix [[1, 1/2], [1/2, 1/3]]:", [1 / e for e in matrix(QQ, [[1, 1/2], [1/2, 1/3]]).eigenvalues()])
