# Exercise exr-lgt-link-u1-difference: the covariant difference of a charged field and its O(a) correction, site and midpoint links.
a, q, x = var('a q x')
A = function('A')(x); phi = function('phi')(x)
D = lambda f: diff(f, x) + I * q * A * f
for name, Ax in (('site', A), ('midpoint', A.subs(x=x + a/2))):
    nabla = (exp(I * a * q * Ax) * phi.subs(x=x + a) - phi) / a
    s = taylor(nabla, a, 0, 1)
    print(name, 'link, order a^0 equals D phi:', bool((s.coefficient(a, 0) - D(phi)).simplify_full() == 0))
    print(name, 'link, order a^1 minus (1/2) D D phi:', (s.coefficient(a, 1) - D(D(phi)) / 2).simplify_full())
