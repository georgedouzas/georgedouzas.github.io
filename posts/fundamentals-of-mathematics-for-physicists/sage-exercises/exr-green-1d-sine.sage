# Exercise exr-green-1d-sine: u = int_0^1 G(x, xi) sin(pi xi) dxi with the tent Green's function equals sin(pi x)/pi^2;
# the Green's function of -u'' - u on [0, 1] from the one-sided solutions sin x and sin(1 - x), and its slope jump.
x, xi = var("x xi")
assume(x > 0, x < 1)
Ga, Gb = x * (1 - xi), xi * (1 - x)                    # x < xi and x > xi
u = integrate(Gb * sin(pi * xi), xi, 0, x) + integrate(Ga * sin(pi * xi), xi, x, 1)
print("u(x) =", u.simplify_full(), "| equals sin(pi x)/pi^2:", bool((u - sin(pi * x) / pi ** 2).simplify_full() == 0))
ua, ub = sin(x), sin(1 - x)                            # solve -u'' - u = 0 with u(0) = 0, u(1) = 0 respectively
W = (ua * diff(ub, x) - diff(ua, x) * ub).simplify_full()
print("Wronskian p(u_a u_b' - u_a' u_b) =", W)
Ha, Hb = ua * ub.subs(x=xi) / (-W), ua.subs(x=xi) * ub / (-W)    # G for x < xi and x > xi
print("G(x, xi) for x < xi:", Ha, "| for x > xi:", Hb)
print("continuous at x = xi:", bool((Ha - Hb).subs(x=xi).simplify_full() == 0))
print("slope jump at x = xi:", (diff(Hb, x) - diff(Ha, x)).subs(x=xi).simplify_full())
print("-G'' - G = 0 on each side and G(0, xi) = G(1, xi) = 0:", bool((-diff(Ha, x, 2) - Ha).simplify_full() == 0), bool((-diff(Hb, x, 2) - Hb).simplify_full() == 0), bool(Ha.subs(x=0) == 0), bool(Hb.subs(x=1).simplify_full() == 0))
