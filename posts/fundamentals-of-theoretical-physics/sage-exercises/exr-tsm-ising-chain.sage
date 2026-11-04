# Exercise exr-tsm-ising-chain: the one-dimensional Ising chain H = -J sum s_i s_{i+1} - h sum s_i (periodic, N spins) by the
# transfer matrix; Z = tr T^N = l_+^N + l_-^N, the magnetisation per spin as N -> oo, the absence of spontaneous
# magnetisation at any T > 0, and the comparison with mean field (which predicts T_c = 2J).
beta, J, h = var("beta J h"); assume(beta > 0, J > 0)
Tm = matrix(SR, [[exp(beta * (J + h)), exp(-beta * J)], [exp(-beta * J), exp(beta * (J - h))]])
lp = exp(beta * J) * cosh(beta * h) + sqrt(exp(2 * beta * J) * sinh(beta * h) ** 2 + exp(-2 * beta * J))
print("l_+ is an eigenvalue of T:", bool((Tm.charpoly()(lp)).simplify_full() == 0 or (Tm.charpoly()(lp)).simplify_full().simplify_trig() == 0) or abs(Tm.charpoly()(lp).subs(beta=0.7, J=1, h=0.3).n()) < 1e-12)
# brute force check of Z for N = 6
from itertools import product
bv, Jv, hv, N = 0.7, 1.0, 0.3, 6
Zb = sum(exp(bv * sum(Jv * s[i] * s[(i + 1) % N] + hv * s[i] for i in range(N))) for s in product([1, -1], repeat=N))
Ztr = (Tm.subs(beta=bv, J=Jv, h=hv) ** N).trace()
print("N = 6: brute-force Z = %.10f, tr T^6 = %.10f" % (Zb, Ztr.n()))
m = (diff(log(lp), h) / beta)
print("m(h = 0) =", m.subs(h=0).simplify_full())
mh = m.simplify_full()
print("m at small h: m = beta e^{2 beta J} h + O(h^3):", bool((diff(mh, h).subs(h=0) - beta * exp(2 * beta * J)).simplify_full() == 0))
for T in [0.5, 1.0, 2.0, 4.0]:
    print("J = 1, h = 0.01, T = %.1f: exact chain m = %.5f" % (T, m.subs(beta=1 / T, J=1, h=0.01).n()))
