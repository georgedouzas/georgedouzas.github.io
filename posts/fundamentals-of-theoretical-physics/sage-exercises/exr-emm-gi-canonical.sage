# Exercise exr-emm-gi-canonical: (x, p) -> (x, p + q grad(lambda)) preserves the Poisson brackets.
var('q')
X0, X1, X2, P0, P1, P2 = var('X0 X1 X2 P0 P1 P2'); Xv = [X0, X1, X2]; Pv = [P0, P1, P2]
lam = function('lam')(X0, X1, X2)
pb = lambda f, g: sum(diff(f, Xv[i]) * diff(g, Pv[i]) - diff(f, Pv[i]) * diff(g, Xv[i]) for i in range(3))
Pn = [Pv[i] + q * diff(lam, Xv[i]) for i in range(3)]
print('{x_i, P_j}:', matrix(3, 3, lambda i, j: pb(Xv[i], Pn[j])))
print('{P_i, P_j}:', matrix(3, 3, lambda i, j: pb(Pn[i], Pn[j]).simplify_full()))
