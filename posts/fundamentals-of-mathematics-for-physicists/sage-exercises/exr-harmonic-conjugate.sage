# Exercise exr-harmonic-conjugate: u = x^3 - 3 x y^2 is harmonic; its harmonic conjugate v with v(0, 0) = 0, found by
# integrating the Cauchy-Riemann equations, gives u + i v = z^3.
x, y = var("x y", domain="real")
u = x^3 - 3*x*y^2
print("Laplacian of u:", (diff(u, x, 2) + diff(u, y, 2)).simplify_full())
v = integrate(diff(u, x), y)                    # v_y = u_x
c = var("c")
cx = integrate(-diff(u, y) - diff(v, x), x)     # v_x = -u_y fixes the x-dependent constant
v = v + cx
print("v =", v, "  v(0,0) =", v.subs(x=0, y=0))
print("Cauchy-Riemann hold:", bool(diff(u, x) == diff(v, y)) and bool(diff(u, y) == -diff(v, x)))
print("u + i v == (x + i y)^3:", bool(((u + I*v) - (x + I*y)^3).expand() == 0))
