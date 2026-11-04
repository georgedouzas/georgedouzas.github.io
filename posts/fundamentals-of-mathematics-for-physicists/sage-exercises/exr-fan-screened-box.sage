# Exercise exr-fan-screened-box: u = (1/2) e^{-|x|} * 1_[-1,1] solves u - u'' = 1_[-1,1]; continuity of u, u' and the jump of u''.
x, y = var('x y')
uin = (integrate(exp(-(x - y)) / 2, y, -1, x) + integrate(exp(-(y - x)) / 2, y, x, 1)).simplify_full()
uout = integrate(exp(-(x - y)) / 2, y, -1, 1).simplify_full()       # x >= 1
print("u inside =", uin, "; equals 1 - e^-1 cosh x:", bool((uin - (1 - exp(-1) * (exp(x) + exp(-x)) / 2)).simplify_full() == 0))
print("u outside (x >= 1) =", uout, "; equals sinh(1) e^-x:", bool((uout - (e - exp(-1)) / 2 * exp(-x)).simplify_full() == 0))
print("u - u'' inside =", (uin - diff(uin, x, 2)).simplify_full(), ", outside =", (uout - diff(uout, x, 2)).simplify_full())
for d in (0, 1, 2):
    print("derivative %d at x = 1: inside %s, outside %s" % (d, diff(uin, x, d).subs(x=1).simplify_full(), diff(uout, x, d).subs(x=1).simplify_full()))
print("jump of u'' at 1:", (diff(uout, x, 2) - diff(uin, x, 2)).subs(x=1).simplify_full())
