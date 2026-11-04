# Exercise exr-fpi-fermion-matsubara: (1/beta) sum_n 1/(((2n+1) pi/beta)^2 + w^2) = tanh(beta w/2)/(2 w) = (1/2 - n_F)/w,
# compared numerically to high precision (Sage returns the sum through digamma functions).
n = var("n"); b, w = var("b w", domain="positive")
S = sum(1 / (((2 * n + 1) * pi / b) ** 2 + w ** 2), n, -oo, oo) / b
for bv, wv in [(1, 1), (2, QQ(1) / 2), (QQ(1) / 3, 4)]:
    print("beta = %s, w = %s:  sum = %s   tanh/(2w) = %s" % (bv, wv, S.subs(b=bv, w=wv).n(80), (tanh(bv * wv / 2) / (2 * wv)).n(80)))
nF = 1 / (exp(b * w) + 1)
print("tanh(b w/2)/(2w) = (1/2 - n_F)/w exactly:", bool((tanh(b * w / 2) / (2 * w) - (QQ(1) / 2 - nF) / w).exponentialize().simplify_full() == 0))
