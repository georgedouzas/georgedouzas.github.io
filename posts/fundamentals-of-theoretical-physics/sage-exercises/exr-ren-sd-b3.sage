# Exercise exr-ren-sd-b3: the beta function after the change of scheme lambda' = lambda + c lambda^2 + d lambda^3.
var('l lp b1 b2 b3 c d')
beta = b1 * l^2 + b2 * l^3 + b3 * l^4
lprime = l + c * l^2 + d * l^3
inv = lp - c * lp^2 + (2 * c^2 - d) * lp^3
print("check of the inverse series:", taylor(lprime.subs(l=inv) - lp, lp, 0, 3))
bp = taylor((beta * diff(lprime, l)).subs(l=inv), lp, 0, 4)
for k in [2, 3, 4]:
    print("coefficient of lambda'^%d:" % k, bp.coefficient(lp, k).expand())
