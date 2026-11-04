# Exercise exr-ren-bf-mom: the momentum-subtraction threshold function f(z) = 6 int_0^1 x^2(1-x)^2 z^2/(1 + x(1-x) z^2) dx.
x = var('x')
f = lambda z: 6 * numerical_integral(x^2 * (1 - x)^2 * z^2 / (1 + x * (1 - x) * z^2), 0, 1)[0]
for z in [0.1, 1, 3, 10, 100]:
    print("f(%s) = %.4f" % (z, f(z)))
print("z^2/5 at z = 0.1: %.4f" % (0.01 / 5))
print("f = 1/2 at z = %.2f,  f = 0.9 at z = %.2f" % (find_root(lambda z: f(z) - 0.5, 0.5, 50), find_root(lambda z: f(z) - 0.9, 1, 100)))
