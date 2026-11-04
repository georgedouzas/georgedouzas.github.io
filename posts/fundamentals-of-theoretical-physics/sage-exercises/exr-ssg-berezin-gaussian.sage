# Exercise exr-ssg-berezin-gaussian: Berezin integration on Sage's exterior algebra. The integral of a function of
# Grassmann variables is the coefficient of the top monomial; for one pair, int dthetabar dtheta exp(-thetabar a theta) = a,
# and for two pairs int exp(-thetabar A theta) = det A, the opposite of the bosonic 1/det A.
E = ExteriorAlgebra(SR, ["t1", "t2", "tb1", "tb2"]); t1, t2, tb1, tb2 = E.gens()
def expo(x, n=4):                                  # exp of a nilpotent even element: the series stops
    return sum((x^k) / factorial(k) for k in range(n + 1))
def berezin(f, order):                             # coefficient of the ordered top monomial
    top = prod(order, E.one())
    (key, sgn), = top.monomial_coefficients().items()
    return f.monomial_coefficients().get(key, 0) * sgn
a = var("a")
f1 = expo(-a * tb1 * t1)
print("exp(-a thetabar theta) =", f1)
print("[PASS] the series stops: exp(-a thetabar theta) = 1 - a thetabar theta:", f1 == 1 - a * tb1 * t1)
print("[PASS] int dthetabar dtheta exp(-a thetabar theta) = a (top monomial t1 tb1):", bool(berezin(f1, [t1, tb1]) == a))
A = matrix(SR, 2, 2, var("A11 A12 A21 A22"))
th = [t1, t2]; thb = [tb1, tb2]
S = -sum(thb[i] * A[i, j] * th[j] for i in range(2) for j in range(2))
f2 = expo(S)
val = berezin(f2, [t1, tb1, t2, tb2])
print("int exp(-thetabar A theta) =", val.expand())
print("[PASS] the Grassmann Gaussian integral is det A:", bool((val - A.det()).expand() == 0))
