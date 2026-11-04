# Exercise exr-emw-zones-crossover: where the radiation term equals the static and induction terms on the equator.
u = var('u', domain='positive')   # u = k r
print('solutions of u^4 = 1 + u^2:', solve(u^4 == 1 + u^2, u), '; numerically', find_root(u^4 - 1 - u^2, 0.5, 3))
