# Exercise exr-nak-inner-sn: the rotation fields K_ij = x_i d_j - x_j d_i of the unit sphere S^n in R^(n+1) have
# |K_ij|^2 = x_i^2 + x_j^2, whose average over S^n is 2/(n+1) (since <x_i^2> = 1/(n+1)): 2/3 on S^2, 1/2 on S^3.
# The averages are computed by integration in hyperspherical coordinates.
def sphere_average(n, f):
    a = [var("a%d" % i) for i in range(n)]            # a_0, ..., a_{n-2} in (0, pi), a_{n-1} in (0, 2 pi)
    x = [prod(sin(a[j]) for j in range(i)) * cos(a[i]) for i in range(n)] + [prod(sin(a[j]) for j in range(n))]
    vol = prod(sin(a[j])^(n - 1 - j) for j in range(n - 1))          # the volume density of the unit S^n
    num = f(x) * vol; den = vol
    for j in range(n - 1):
        num = integrate(num, a[j], 0, pi); den = integrate(den, a[j], 0, pi)
    num = integrate(num, a[n - 1], 0, 2 * pi); den = integrate(den, a[n - 1], 0, 2 * pi)
    return (num / den).simplify_full(), x
for n in [2, 3, 4]:
    v, x = sphere_average(n, lambda x: x[0]^2 + x[1]^2)
    print("n = %d: <|K_01|^2> = %s, and x.x = %s" % (n, v, sum(xi^2 for xi in x).simplify_full()))
    print("[PASS] on S^%d the average of |K_ij|^2 is 2/(n+1) = %s:" % (n, 2 / (n + 1)), bool(v == 2 / (n + 1)))
