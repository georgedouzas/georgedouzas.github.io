# Exercise exr-acsym-complex-quartic: the complex scalar with quartic self-interaction in 1+1 dimensions (signature
# (+, -)), L = d phibar . d phi - m^2 |phi|^2 - lam |phi|^4, written with phi = u + i v. Euler-Lagrange expressions,
# conservation of the U(1) current j = 2(v du - u dv) and of T^{mu nu} = d^mu phibar d^nu phi + d^nu phibar d^mu phi
# - eta^{mu nu} L as identities in the Euler-Lagrange expressions, positivity of T^{tt}, and the rotating solution.
t, x = var('t x', domain='real'); X = [t, x]; m, lam = var('m lam', domain='positive')
eta = diagonal_matrix(SR, [1, -1])
pu, pv = var('pu pv')                                                    # jet variables for the values of u and v
u, v = function('u')(t, x), function('v')(t, x)
Vjet = m^2 * (pu^2 + pv^2) + lam * (pu^2 + pv^2)^2                       # V = m^2 |phi|^2 + lam |phi|^4
V = Vjet.subs({pu: u, pv: v})
du = [diff(u, X[mu]) for mu in range(2)]; dv = [diff(v, X[mu]) for mu in range(2)]
up = lambda w, mu: eta[mu, mu] * w[mu]                                   # raise the index
L = sum(eta[mu, mu] * (du[mu]^2 + dv[mu]^2) for mu in range(2)) - V     # d phibar . d phi = (du)^2 + (dv)^2
box = lambda f: diff(f, t, 2) - diff(f, x, 2)
# Euler-Lagrange expressions d_mu dL/d(d_mu u) - dL/du, with dL/d(d_mu u) = 2 d^mu u
Eu = 2 * box(u) + diff(Vjet, pu).subs({pu: u, pv: v}); Ev = 2 * box(v) + diff(Vjet, pv).subs({pu: u, pv: v})
print("E_u =", Eu.expand())
j = [2 * (v * up(du, mu) - u * up(dv, mu)) for mu in range(2)]
divj = sum(diff(j[mu], X[mu]) for mu in range(2))
print("d_mu j^mu = v E_u - u E_v identically:", bool((divj - (v * Eu - u * Ev)).simplify_full() == 0))
T = matrix(SR, 2, 2, lambda mu, nu: 2 * (up(du, mu) * up(du, nu) + up(dv, mu) * up(dv, nu)) - eta[mu, nu] * L)
for nu in range(2):
    div = sum(diff(T[mu, nu], X[mu]) for mu in range(2))
    print("nu = %s: d_mu T^{mu nu} = E_u d^nu u + E_v d^nu v identically:" % X[nu], bool((div - (Eu * up(du, nu) + Ev * up(dv, nu))).simplify_full() == 0))
print("T^{tt} =", T[0, 0].expand(), " (a sum of squares plus V >= 0)")
A, w = var('A w', domain='positive')
Eus = (2 * box(A * cos(w * t)) + diff(Vjet, pu).subs({pu: A * cos(w * t), pv: -A * sin(w * t)})).simplify_full()
print("for phi = A exp(-i w t): E_u =", Eus.factor(), "  vanishes iff w^2 = m^2 + 2 lam A^2")
us, vs = A * cos(w * t), -A * sin(w * t)
j0 = 2 * (vs * diff(us, t) - us * diff(vs, t))
print("its charge density j^0 =", j0.simplify_full(), "(constant), and T^{tt} =", (diff(us, t)^2 + diff(vs, t)^2 + Vjet.subs({pu: us, pv: vs})).simplify_full())
