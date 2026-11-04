# Exercise exr-modulus-circle-equation: the locus |z - 1| = 2 |z + 1| is a circle; square both sides, expand and
# complete the square to read off centre and radius.
x, y = var("x y", domain="real")
eq = ((x - 1)^2 + y^2) - 4*((x + 1)^2 + y^2)      # |z-1|^2 - 4 |z+1|^2 = 0
eq = (eq / (-3)).expand()                          # normalise the quadratic terms to x^2 + y^2
print("x^2 + y^2 + ... = 0 with:", eq, "= 0")
cx = -eq.coefficient(x, 1)/2
r2 = cx^2 - eq.subs(x=0, y=0)
print("centre:", (cx, 0), " radius:", sqrt(r2))
print("check: the circle equation reproduces the locus:", bool((((x - cx)^2 + y^2 - r2) - eq).expand() == 0))
