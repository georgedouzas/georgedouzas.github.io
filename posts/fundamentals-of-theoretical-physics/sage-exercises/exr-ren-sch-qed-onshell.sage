# Exercise exr-ren-sch-qed-onshell: 1/alpha at the Z mass with the electron alone, in MS-bar and as the one-loop
# effective charge.
al = 1 / 137.035999; me = 0.000510999; MZ = 91.1876
x = var('x')
I = numerical_integral(x * (1 - x) * log(1 + x * (1 - x) * MZ^2 / me^2), 0, 1)[0]
ms = 1 / al - 2 / (3 * pi) * log(MZ / me)
eff = (1 - 2 * al / pi * I) / al
ll = 1 / al - 1 / (3 * pi) * (log(MZ^2 / me^2) - 5 / 3)
print("1/alpha_MSbar(MZ) = %.4f   1/alpha_eff(MZ) = %.4f   large-Q formula %.4f" % (ms.n(), eff.n(), ll.n()))
print("difference eff - MSbar = %.4f, 5/(9 pi) = %.4f" % ((eff - ms).n(), (5 / (9 * pi)).n()))
