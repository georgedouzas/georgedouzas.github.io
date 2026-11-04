# Exercise exr-ren-sd-omega: the slope of the beta function at its zero is unchanged by lambda' = lambda + c lambda^2.
var('l lp eps b1 b2 c')
beta = -eps * l + b1 * l^2 + b2 * l^3
lam_of = (-1 + sqrt(1 + 4 * c * lp)) / (2 * c)                 # inverse of lp = l + c l^2
betap = (beta * (1 + 2 * c * l)).subs(l=lam_of)
for (e, B1, B2, C) in [(0.1, 3, -17 / 3, -3), (0.5, 1, 2, 0.7)]:
    ls = [r for r in solve(beta.subs(eps=e, b1=B1, b2=B2) / l == 0, l) if r.rhs().n() > 0][0].rhs().n()
    lps = ls + C * ls^2
    sp = diff(betap, lp).subs(eps=e, b1=B1, b2=B2, c=C, lp=lps).n()
    s0 = diff(beta, l).subs(eps=e, b1=B1, b2=B2, l=ls).n()
    print("eps = %.1f: lambda* = %.6f, beta' at lambda'* = %.2e, slopes %.10f and %.10f" % (e, ls, betap.subs(eps=e, b1=B1, b2=B2, c=C, lp=lps).n(), sp, s0))
