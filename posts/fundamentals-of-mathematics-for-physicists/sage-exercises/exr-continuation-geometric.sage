# Exercise exr-continuation-geometric: closed forms of sum z^n and sum (n + 1) z^n, which continue the series beyond
# the unit disc, evaluated at z = 2 and z = -3.
z, n = var("z n")
assume(abs(z) < 1)
s1 = sum(z^n, n, 0, oo)
s2 = sum((n + 1)*z^n, n, 0, oo)
print("sum z^n =", s1.simplify_full(), "   sum (n+1) z^n =", s2.simplify_full())
forget()
for w in (2, -3):
    print("at z = %s: %s, %s" % (w, s1.subs(z=w), s2.subs(z=w)))
print("the series itself diverges at z = 2: partial sums", [sum(2^k for k in range(N)) for N in range(1, 7)])
