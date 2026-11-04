# Exercise exr-ren-ldse-six: gamma = (sqrt(5 + 4 sqrt(1 - a)) - 3)/2 satisfies gamma(gamma+1)(gamma+2)(gamma+3) = -a.
var('a')
g = (sqrt(5 + 4 * sqrt(1 - a)) - 3) / 2
print("gamma(gamma+1)(gamma+2)(gamma+3) + a simplifies to:", (g * (g + 1) * (g + 2) * (g + 3) + a).simplify_full())
print("series:", taylor(g, a, 0, 3))
