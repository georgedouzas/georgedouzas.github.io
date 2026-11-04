# Exercise exr-fan-volterra-powers: iterating V f(x) = int_0^x f on f = 1 and on a general kernel; |V^n| = V^n 1 (1) = 1/n!.
x, t, s = var('x t s')
f = function('f')
Vn = lambda n: (x - t)^(n - 1) / factorial(n - 1)
ok = all(bool((integrate(Vn(n).subs(x=s), s, t, x) - Vn(n + 1)).simplify_full() == 0) for n in range(1, 8))
print("int_t^x (s - t)^(n-1)/(n-1)! ds = (x - t)^n/n! for n = 1..7:", ok)
g = 1
for n in range(1, 7):
    g = integrate(g.subs(x=t) if g != 1 else 1, t, 0, x)
    print("V^%d 1 = %s, norm 1/%d! = %s, |V^n|^(1/n) = %.4f" % (n, g, n, 1 / factorial(n), (1 / factorial(n))^(1 / n).n()))
