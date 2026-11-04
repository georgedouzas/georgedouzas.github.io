# Exercise exr-qftx-zero-dim-diagrams: <phi^2> and <phi^4> in the zero-dimensional quartic theory to order lambda^2 from
# Gaussian moments, the diagram sum 1/4 + 1/4 + 1/6 = 2/3, and the Schwinger-Dyson relation <phi^2> + (lambda/6)<phi^4> = 1.
l = var("l")
mom = lambda k: prod(range(k - 1, 0, -2)) if k % 2 == 0 else 0
def avg(k, order=3):
    return sum((-l / 24) ** n / factorial(n) * mom(k + 4 * n) for n in range(order))
Z = avg(0)
p2 = (avg(2) / Z).series(l, 3).truncate(); p4 = (avg(4) / Z).series(l, 3).truncate()
print("<phi^2> =", p2, "   <phi^4> =", p4)
print("diagrams at order lambda^2: 1/4 + 1/4 + 1/6 =", QQ(1) / 4 + QQ(1) / 4 + QQ(1) / 6)
print("Schwinger-Dyson <phi^2> + (lambda/6)<phi^4> - 1 =", (p2 + l / 6 * p4 - 1).expand().series(l, 3).truncate())
