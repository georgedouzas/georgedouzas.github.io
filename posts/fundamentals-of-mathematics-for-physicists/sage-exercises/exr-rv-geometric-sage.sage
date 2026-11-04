# Exercise exr-rv-geometric-sage: the geometric distribution P(K = k) = (1 - p)^(k-1) p, k >= 1: total probability,
# mean 1/p, variance (1 - p)/p^2 and characteristic function p e^{it}/(1 - (1 - p) e^{it}); p = 1/2 recovers E K = 2.
k, p, t = var("k p t")
assume(k, "integer"); assume(p > 0); assume(p < 1)
pk = (1 - p) ** (k - 1) * p
tot = SR(sum(pk, k, 1, oo)).simplify_full()
m1 = SR(sum(k * pk, k, 1, oo)).simplify_full()
m2 = SR(sum(k ** 2 * pk, k, 1, oo)).simplify_full()
cf = SR(sum(exp(I * t * k) * pk, k, 1, oo)).simplify_full()
print("total probability:", tot)
print("mean:", m1, "; variance:", (m2 - m1 ** 2).simplify_full())
print("characteristic function:", cf, "; equals p e^{it}/(1 - (1-p) e^{it}):", bool((cf - p * exp(I * t) / (1 - (1 - p) * exp(I * t))).simplify_full() == 0))
print("at p = 1/2: mean", m1.subs(p=QQ(1) / 2), ", variance", (m2 - m1 ** 2).simplify_full().subs(p=QQ(1) / 2), ", E K^2 =", m2.subs(p=QQ(1) / 2))
