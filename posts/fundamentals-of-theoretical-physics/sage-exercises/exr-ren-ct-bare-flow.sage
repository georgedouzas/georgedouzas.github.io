# Exercise exr-ren-ct-bare-flow: the bare relation lambda_0 mu^-eps = lambda/(1 - b lambda/eps), its exact beta function
# -eps lambda + b lambda^2, the solution with lambda(mu_0) = lambda_1, and its limit eps -> 0.
var('lam eps t lam1 b')
bare = lam / (1 - b * lam / eps)
print("poles:", taylor(bare, lam, 0, 4))
beta = (-eps * bare / diff(bare, lam)).simplify_full()
print("beta =", beta.expand())
lstar = eps / b
sol = lstar / (1 - (1 - lstar / lam1) * exp(eps * t))
print("solves the flow:", bool((diff(sol, t) - (-eps * sol + b * sol^2)).simplify_full() == 0), "  initial value:", sol.subs(t=0).simplify_full())
print("eps -> 0 limit:", limit(sol, eps=0).simplify_full())
bv = 3 / (16 * pi^2)
for e in [0.03, 0.06]:
    print("eps = %.2f, lambda_1 = 2: lambda* = %.3f, lambda(t = -200) = %.3f" % (e, (e / bv).n(), sol.subs(eps=e, b=bv, lam1=2, t=-200).n()))
