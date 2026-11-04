# Exercise exr-ren-sch-mom: the conversion lambda_MOM = lambda_MSbar - 3 lambda^2/(16 pi^2) at the symmetric point.
var('x lam')
c = -integrate(log(x * (1 - x)), x, 0, 1)
print("16 pi^2 B_R at Q = mu (massless):", c)
lmom = lam - lam^2 / 2 * 3 * c / (16 * pi^2)
print("lambda_MOM =", lmom)
print("for lambda_MSbar = 3: lambda_MOM = %.4f, difference %.4f (%.1f%%)" % (lmom.subs(lam=3).n(), (3 - lmom.subs(lam=3)).n(), (100 * (3 - lmom.subs(lam=3)) / 3).n()))
