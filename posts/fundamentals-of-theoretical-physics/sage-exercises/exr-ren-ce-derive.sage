# Exercise exr-ren-ce-derive: critical exponents of the O(n) model to order eps^2 from the two-loop minimal-subtraction
# functions beta = -eps u + (n+8)/3 u^2 - (3n+14)/3 u^3, gamma_phi = (n+2)/36 u^2, gamma_m2 = (n+2)/3 u - 5(n+2)/18 u^2.
var('u e n a1 a2')
beta = -e * u + (n + 8) / 3 * u^2 - (3 * n + 14) / 3 * u^3
gm2 = (n + 2) / 3 * u - 5 * (n + 2) / 18 * u^2; gphi = (n + 2) / 36 * u^2
us = a1 * e + a2 * e^2
eq = taylor(beta.subs(u=us), e, 0, 3)
sol = [s for s in solve([eq.coefficient(e, 2) == 0, eq.coefficient(e, 3) == 0], [a1, a2], solution_dict=True) if s[a1] != 0][0]
ustar = us.subs(sol)
print("u* =", sol[a1], "* eps +", sol[a2].factor(), "* eps^2")
nu = taylor(1 / (2 - gm2.subs(u=ustar)), e, 0, 2); eta = taylor(2 * gphi.subs(u=ustar), e, 0, 2); om = taylor(diff(beta, u).subs(u=ustar), e, 0, 2)
print("nu: eps coefficient", nu.coefficient(e, 1).factor(), "  eps^2 coefficient", nu.coefficient(e, 2).factor())
print("eta =", eta.factor(), "   omega =", om.collect(e))
print("n = 1: nu =", nu.subs(n=1), "  eta =", eta.subs(n=1))
