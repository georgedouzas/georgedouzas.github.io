# Exercise exr-ren-x-leading-logs: the L^2 coefficient of the renormalised two-loop symmetric-point vertex function from
# the chain (B - K)^2 and the eye (1 - K)(eye - K B), units (16 pi^2)^-2, and the leading logarithms of lambda-bar.
var('e L lam t b'); d = 4 - e
G = lambda p, q: gamma(p + q - d / 2) * gamma(d / 2 - p) * gamma(d / 2 - q) / (gamma(p) * gamma(q) * gamma(d - p - q))
B = exp(euler_gamma * e / 2) * G(1, 1) * exp(-e * L / 2)
eye = exp(euler_gamma * e) * G(1, 1) * G(1 + e / 2, 1) * exp(-e * L)
chainR = ((B - 2 / e)^2).series(e, 1).truncate().coefficient(e, 0).expand()
eyeS = (eye - (2 / e) * B).series(e, 1).truncate().expand(); eyeR = eyeS.coefficient(e, 0).expand()
print("renormalised chain:", chainR, "   renormalised eye:", eyeR)
L2 = 3 * QQ(1/4) * chainR.coefficient(L, 2) + 6 * QQ(1/2) * eyeR.coefficient(L, 2)
print("L^2 coefficient of the two-loop vertex function:", L2, " -> with L = 2t:", 4 * L2, "t^2  (b^2 (16 pi^2)^2 = 9)")
print("expansion of lambda-bar:", taylor(lam / (1 - b * lam * t), lam, 0, 4))
