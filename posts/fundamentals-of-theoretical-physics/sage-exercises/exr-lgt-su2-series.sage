# Exercise exr-lgt-su2-series: small- and large-beta expansions of u(beta) = I_2(beta)/I_1(beta).
b = var('b')
u = bessel_I(2, b) / bessel_I(1, b)
print('small beta:', taylor(u, b, 0, 5))
for bb in (10, 50, 200):
    print('beta = %d: u = %.6f, 1 - 3/(2 beta) = %.6f, 1 - 3/(2 beta) + 3/(8 beta^2) = %.6f' % (bb, u(b=bb).n(), 1 - 1.5 / bb, 1 - 1.5 / bb + 3 / (8 * bb^2)))
x = var('x')
asym = lambda nu: 1 - (4*nu^2 - 1) / (8*x) + (4*nu^2 - 1) * (4*nu^2 - 9) / (128*x^2)
print('ratio of the asymptotic series:', taylor((asym(2) / asym(1)).subs(x=1/b), b, 0, 2).subs(b=1/x))
