# Exercise exr-cr-check: the Cauchy-Riemann equations for z^3, 1/z and cos z, and f' = u_x + i v_x.
x, y, z = var("x y z"); assume(x, "real"); assume(y, "real")
def check(f):
    g = f.subs(z=x + I*y).expand()
    u, v = g.real_part().simplify_full(), g.imag_part().simplify_full()
    cr = bool((diff(u, x) - diff(v, y)).simplify_full() == 0) and bool((diff(u, y) + diff(v, x)).simplify_full() == 0)
    fp = (diff(u, x) + I*diff(v, x)).simplify_full()
    return cr, fp
for f in (z^3, 1/z, cos(z)):
    cr, fp = check(f)
    print("f =", f, ": Cauchy-Riemann hold:", cr, "   u_x + i v_x =", fp, "  equals f'(z):", bool((fp - diff(f, z).subs(z=x + I*y)).simplify_full() == 0))
