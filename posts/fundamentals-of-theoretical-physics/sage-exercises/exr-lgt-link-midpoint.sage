# Exercise exr-lgt-link-midpoint: the link angle a A(midpoint) reproduces the line integral of A to O(a^3); a A(site) only to O(a^2).
a, x = var('a x')
A = function('A')(x)
t = var('t')
line = integrate(taylor(A.subs(x=x + t), t, 0, 3), t, 0, a)
mid = taylor(a * A.subs(x=x + a/2), a, 0, 3)
site = a * A
print('line integral - a A(x + a/2) =', (line - mid).expand().simplify_full())
print('line integral - a A(x)       =', taylor(line - site, a, 0, 3).simplify_full())
