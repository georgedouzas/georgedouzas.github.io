# Exercise exr-ren-pv-tadpole: the Pauli-Villars tadpole with regulators of masses M and sqrt(2) M, exactly and as a
# numerical integral with a large cutoff (units 16 pi^2 T).
var('m M c1 c2', domain='positive')
sol = solve([c1 + c2 == 1, c1 * M^2 + 2 * c2 * M^2 == m^2], c1, c2, solution_dict=True)[0]
print("c1 =", sol[c1], "  c2 =", sol[c2])
T = m^2 * log(m^2) - sol[c1] * M^2 * log(M^2) - sol[c2] * 2 * M^2 * log(2 * M^2)
claim = 2 * log(2) * M^2 - m^2 * log(4 * M^2 / m^2)
print("16 pi^2 T_PV - claim simplifies to 0:", bool((T - claim).simplify_log().simplify_full() == 0))
mv, Mv = 1, 10
c1v, c2v = (sol[c1].subs(m=mv, M=Mv), sol[c2].subs(m=mv, M=Mv))
f = lambda l: 2 * l^3 * (1 / (l^2 + mv^2) - c1v / (l^2 + Mv^2) - c2v / (l^2 + 2 * Mv^2))
num = numerical_integral(f, 0, 10^5)[0]
print("numerical 16 pi^2 T_PV (cutoff 1e5): %.6f   formula: %.6f" % (num, claim.subs(m=mv, M=Mv).n()))
