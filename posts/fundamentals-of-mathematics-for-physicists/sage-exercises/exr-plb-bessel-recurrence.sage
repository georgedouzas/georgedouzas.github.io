# Exercise exr-plb-bessel-recurrence: J_{n-1} + J_{n+1} - (2n/x) J_n and J_{n-1} - J_{n+1} - 2 J_n' evaluated for
# n = 1, 2, 3 at x = 0.7, 1.7, 5.3 to 50 digits.
x = var("x")
R = RealField(200)
ok = True
for n in range(1, 4):
    for xv in [R("0.7"), R("1.7"), R("5.3")]:
        r1 = bessel_J(n - 1, xv) + bessel_J(n + 1, xv) - 2 * n / xv * bessel_J(n, xv)
        r2 = (bessel_J(n - 1, x) - bessel_J(n + 1, x) - 2 * diff(bessel_J(n, x), x)).subs(x=xv)
        ok = ok and abs(r1) < R(10) ** (-45) and abs(R(r2)) < R(10) ** (-45)
        print("n = %d, x = %.1f: |sum relation| = %.1e, |derivative relation| = %.1e" % (n, float(xv), float(abs(r1)), float(abs(R(r2)))))
print("all residuals below 1e-45:", ok)
