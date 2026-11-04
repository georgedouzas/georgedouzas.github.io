# Exercise exr-vcp-fluxpot-kklt: the KKLT model with K = -3 log(T + Tbar) and W = W0 + B exp(-2 pi T) (Grana, eqs. 7.7-
# 7.8). On the real line T = sigma, D_T W = d_T W + (d_T K) W = 0 gives W0 = -B e^{-2 pi sigma}(1 + 4 pi sigma/3).
# For W0 = -10^-4, B = 1 the solution is sigma ~ 1.8; the AdS value is V = -3 e^K |W|^2 = -2 pi^2 B^2 e^{-4 pi sigma}/(3 sigma)
# (Grana, eq. 7.9).
s, W0, B = var("sigma W0 B", domain="real")
K = -3 * log(2 * s)
W = W0 + B * exp(-2 * pi * s)
DW = diff(W, s) + diff(K, s) / 2 * W           # d/dT of K(T + Tbar) at real T: d_T K = -3/(T + Tbar) = -3/(2 sigma)
sol = solve(DW == 0, W0)[0].rhs()
print("D_T W = 0  =>  W0 =", sol.simplify_full())
print("agrees with W0 = -B e^{-2 pi sigma}(1 + 4 pi sigma/3):", bool((sol + B * exp(-2 * pi * s) * (1 + 4 * pi * s / 3)).simplify_full() == 0))
Vads = (-3 * exp(K) * W**2).subs(W0=sol).simplify_full()
print("V = -3 e^K W^2 at the minimum =", Vads, "; equals -2 pi^2 B^2 e^{-4 pi sigma}/(3 sigma):",
      bool((Vads + 2 * pi**2 * B**2 * exp(-4 * pi * s) / (3 * s)).simplify_full() == 0))
sc = find_root(sol.subs(B=1) + 1e-4, 0.1, 10)
print("W0 = -1e-4, B = 1: sigma_cr = %.4f ; V_min = %.3e" % (sc, -2 * RR(pi)**2 * RR(exp(-4 * RR(pi) * RR(sc))) / (3 * RR(sc))))
for w0 in [-1e-4, -1e-8, -1e-12]:
    print("W0 =", w0, ": sigma_cr =", "%.3f" % find_root(sol.subs(B=1) - w0, 0.01, 50), "(grows like log 1/|W0|)")
