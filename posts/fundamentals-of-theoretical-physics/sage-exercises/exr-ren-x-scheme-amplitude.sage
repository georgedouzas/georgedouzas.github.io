# Exercise exr-ren-x-scheme-amplitude: the symmetric-point amplitude M(Q) of the massless quartic theory to one loop
# from lambda_MSbar(mu) and from lambda_MOM(mu); they agree to order lambda^2.
var('lam L')        # L = log(Q/mu)
b = 3 / (16 * pi^2)
M_ms = -lam + 3 * lam^2 / (32 * pi^2) * (2 - 2 * L)
lmom = lam - b * lam^2
M_mom = -lmom - b * lmom^2 * L
print("difference to order lambda^2:", taylor(M_ms - M_mom, lam, 0, 2).simplify_full())
print("difference at order lambda^3:", taylor(M_ms - M_mom, lam, 0, 3).coefficient(lam, 3).simplify_full(), "* lambda^3")
v = {lam: 2, L: log(10)}
print("lambda = 2, Q = 10 mu: M_MSbar = %.5f, M_MOM = %.5f, difference %.5f" % (M_ms.subs(v).n(), M_mom.subs(v).n(), (M_ms - M_mom).subs(v).n()))
