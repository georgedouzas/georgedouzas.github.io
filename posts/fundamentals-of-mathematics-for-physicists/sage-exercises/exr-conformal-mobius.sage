# Exercise exr-conformal-mobius: the Cayley map f(z) = (z - i)/(z + i) sends the real line to the unit circle and
# the upper half-plane to the disc; its inverse, and the images of i, 0, 1 and infinity.
z, w = var("z w")
f = (z - I)/(z + I)
x = var("x", domain="real")
print("|f(x)|^2 for real x:", (f.subs(z=x)*conjugate(f.subs(z=x))).simplify_full())
inv = solve(w == f, z)[0].rhs()
print("inverse:", inv.simplify_full())
print("f(inverse(w)) == w:", bool((f.subs(z=inv) - w).simplify_full() == 0))
for p in (I, 0, 1):
    print("f(%s) = %s" % (p, f.subs(z=p).simplify_full()))
print("limit at infinity:", limit(f, z=oo))
print("Im inverse(w) for w = 1/2:", (inv.subs(w=1/2)).imag_part(), " > 0")
