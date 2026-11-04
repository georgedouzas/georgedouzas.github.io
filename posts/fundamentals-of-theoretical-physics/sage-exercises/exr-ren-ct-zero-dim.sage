# Exercise exr-ren-ct-zero-dim: moments of exp(-phi^2/2 - lam phi^4/24) to order lam^2, the renormalised coupling
# g = -<phi^4>_c/<phi^2>^2 and the connected six-point function as a series in g.
var('lam g')
mom = lambda k: 0 if k % 2 else prod(range(k - 1, 0, -2))      # Gaussian moments (k-1)!!
def avg(k, order=3):
    w = sum((-lam / 24)^j / factorial(j) * mom(k + 4 * j) for j in range(order + 1))
    z = sum((-lam / 24)^j / factorial(j) * mom(4 * j) for j in range(order + 1))
    return taylor(w / z, lam, 0, order)
p2, p4, p6 = avg(2), avg(4), avg(6)
c4 = taylor(p4 - 3 * p2^2, lam, 0, 3)
c6 = taylor(p6 - 15 * p4 * p2 + 30 * p2^3, lam, 0, 3)
print("<phi^2> =", taylor(p2, lam, 0, 2), "   <phi^4> =", taylor(p4, lam, 0, 2))
print("<phi^4>_c =", taylor(c4, lam, 0, 2), "   <phi^6>_c =", taylor(c6, lam, 0, 3))
gl = taylor(-c4 / p2^2, lam, 0, 3)
print("g(lam) =", gl)
inv = taylor(g + sum(var('k%d' % i) * g^i for i in [2, 3]), g, 0, 3)
sol = solve([taylor(gl.subs(lam=inv) - g, g, 0, 3).coefficient(g, i) == 0 for i in [2, 3]], [var('k2'), var('k3')], solution_dict=True)[0]
lam_g = inv.subs(sol)
print("lam(g) =", lam_g)
print("<phi^6>_c in terms of g:", taylor(c6.subs(lam=lam_g), g, 0, 3))
