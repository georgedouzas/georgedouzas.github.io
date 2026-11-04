# Exercise exr-yang-mills-equations: the SU(2) field strength for a potential with derivatives and the Yang-Mills
# equations D_mu F^{mu nu} = 0 in 1+1 dimensions with Sage: a pure-gauge potential A = g^{-1} d g has F = 0.
t, x = var("t x", domain="real")
s = [matrix(SR, [[0, 1], [1, 0]]), matrix(SR, [[0, -I], [I, 0]]), matrix(SR, [[1, 0], [0, -1]])]
T = [-I / 2 * sa for sa in s]
X = [t, x]
def F_of(A):            # F_{mu nu} = d_mu A_nu - d_nu A_mu + [A_mu, A_nu]
    return [[(diff(A[nu], X[mu]) - diff(A[mu], X[nu]) + A[mu] * A[nu] - A[nu] * A[mu]).simplify_full() for nu in range(2)] for mu in range(2)]
f = function("f")(t, x)
g = cos(f / 2) * identity_matrix(SR, 2) - I * sin(f / 2) * s[2]          # g = exp(f T_3)
ginv = cos(f / 2) * identity_matrix(SR, 2) + I * sin(f / 2) * s[2]            # g^{-1} = exp(-f T_3)
A_pure = [(ginv * diff(g, X[mu])).apply_map(lambda e: e.simplify_full()) for mu in range(2)]
F_pure = F_of(A_pure)
print("pure gauge A = g^{-1} dg with g = exp(f(t,x) T_3): A_mu =", [a.list() for a in A_pure])
print("its field strength vanishes:", all(bool(F_pure[mu][nu].apply_map(lambda e: e.simplify_full()) == 0) for mu in range(2) for nu in range(2)))
c1, c2 = var("c1 c2", domain="real")
A_const = [c1 * T[0], c2 * T[1]]
F_const = F_of(A_const)
print("constant potential A_t = c1 T_1, A_x = c2 T_2: F_tx =", F_const[0][1].list(), "= c1 c2 T_3:", bool(F_const[0][1] == c1 * c2 * T[2]))
# Yang-Mills equation D_mu F^{mu nu} = d_mu F^{mu nu} + [A_mu, F^{mu nu}] for the constant potential (eta = diag(1, -1))
eta = [1, -1]
Fup = [[eta[mu] * eta[nu] * F_const[mu][nu] for nu in range(2)] for mu in range(2)]
YM = [sum(diff(Fup[mu][nu], X[mu]) + A_const[mu] * Fup[mu][nu] - Fup[mu][nu] * A_const[mu] for mu in range(2)) for nu in range(2)]
print("Yang-Mills equations D_mu F^{mu nu} for the constant potential:", [y.list() for y in YM], " -> not zero: a constant non-abelian potential is not a solution unless c1 c2 = 0")
