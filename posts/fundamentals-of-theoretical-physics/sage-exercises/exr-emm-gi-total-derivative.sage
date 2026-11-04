# Exercise exr-emm-gi-total-derivative: under a gauge transformation the Lagrangian changes by q d(lambda)/dt along the path.
var('m q t')
X0, X1, X2 = var('X0 X1 X2'); V0, V1, V2 = var('V0 V1 V2'); Xv = [X0, X1, X2]; Vv = [V0, V1, V2]
phi = function('phi')(t, X0, X1, X2); A = [function('A%d' % i)(t, X0, X1, X2) for i in range(3)]; lam = function('lam')(t, X0, X1, X2)
Lf = lambda ph, AA: m / 2 * sum(v^2 for v in Vv) - q * ph + q * sum(Vv[i] * AA[i] for i in range(3))
L1 = Lf(phi, A); L2 = Lf(phi - diff(lam, t), [A[i] + diff(lam, Xv[i]) for i in range(3)])
xs = [function('x%d' % i)(t) for i in range(3)]
path = {**{Xv[i]: xs[i] for i in range(3)}, **{Vv[i]: diff(xs[i], t) for i in range(3)}}
print('L2 - L1 - q d/dt lambda(t, x(t)) =', ((L2 - L1).subs(path) - q * diff(lam.subs(path), t)).simplify_full())
