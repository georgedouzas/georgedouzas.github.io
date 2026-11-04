# Exercise exr-fan-wiener-norms: a = delta0 + delta1 - delta2 in l^1(Z); its autocorrelation a* a, norms, and |a^|^2.
a = {0: 1, 1: 1, 2: -1}
ac = {}
for m, am in a.items():
    for mm, amm in a.items():
        ac[mm - m] = ac.get(mm - m, 0) + am * amm          # (a* a)_n = sum_m conj(a_{m-n}) a_m
print("a* a =", dict(sorted(ac.items())), "; l^1 norm:", sum(abs(v) for v in ac.values()))
th = var('th')
ahat = sum(v * exp(I * k * th) for k, v in a.items())
mod2 = (ahat * ahat.conjugate()).simplify_full().simplify_trig()
print("|a^(theta)|^2 =", mod2.reduce_trig(), "; max over theta:", max(mod2.subs(th=t).n() for t in [k * pi / 200 for k in range(401)]))
print("|a^|_inf = sqrt(5) = %.5f < |a|_1 = 3" % sqrt(5).n())
