# Exercise exr-spd-ahat-expansion: the series (x/2)/sinh(x/2) and the parts of degree 2 and 4 of the product over two roots
# in terms of p1 = x1^2 + x2^2 and p2 = x1^2 x2^2.
x, x1, x2, P1, P2 = var("x x1 x2 P1 P2")
f = (x / 2) / sinh(x / 2)
print("(x/2)/sinh(x/2) =", f.taylor(x, 0, 6))
prod2 = (f.subs(x=x1) * f.subs(x=x2)).taylor((x1, 0), (x2, 0), 4).expand()
part = lambda d: sum(prod2.coefficient(x1, i).coefficient(x2, d - i) * x1 ^ i * x2 ^ (d - i) for i in range(d + 1))
p1, p2 = x1 ^ 2 + x2 ^ 2, x1 ^ 2 * x2 ^ 2
print("degree 2:", part(2), "= -p1/24:", bool((part(2) + p1 / 24).expand() == 0))
print("degree 4:", part(4), "= (7 p1^2 - 4 p2)/5760:", bool((part(4) - (7 * p1 ^ 2 - 4 * p2) / 5760).expand() == 0))
