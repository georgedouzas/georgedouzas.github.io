# Exercise exr-picard-iterates: Picard iterates of y' = x + y, y(0) = 1, compared with the Taylor polynomials of the
# exact solution y = 2 e^x - x - 1.
x, s = var("x s")
exact = 2 * exp(x) - x - 1
print("exact solution solves the problem:", bool((diff(exact, x) - x - exact).simplify_full() == 0) and bool(exact.subs(x=0) == 1))
phi = SR(1)
for k in range(1, 5):
    phi = 1 + integrate(s + phi.subs(x=s), s, 0, x)
    taylor_k = exact.taylor(x, 0, k)
    print("iterate", k, ":", phi.expand(), "| its terms up to degree %d are the Taylor polynomial %s:" % (k, taylor_k), bool((phi.taylor(x, 0, k) - taylor_k).expand() == 0))
