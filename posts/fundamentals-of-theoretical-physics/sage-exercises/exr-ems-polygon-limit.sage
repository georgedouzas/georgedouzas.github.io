# Exercise exr-ems-polygon-limit (Macchi et al., Problem 5.13): field at the centre of a regular n-gon over that of the circle.
n = var('n')
ratio = n * tan(pi / n) / pi
print("expansion for large n:", ratio.subs(n=1 / var('u')).taylor(var('u'), 0, 4))
first = min(k for k in range(3, 200) if (k * tan(pi / k) / pi).n() - 1 < 0.01)
print("smallest n with excess below 1%%: %d (excess %.5f; n - 1 gives %.5f)" % (first, (first * tan(pi / first) / pi).n() - 1, ((first - 1) * tan(pi / (first - 1)) / pi).n() - 1))
