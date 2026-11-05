# name: fa-ft-deriv-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates with rigorous ball integration
# goal: sage_test KaluzaKlein.FourierAnalysis.TheTransform.ft_deriv file "fa-ft-deriv-test.sage"
# description: no recipe certifies the derivative rule for every integrable differentiable f; at f = e^{-x²} and
#   f = x e^{-x²} Sage integrates f'(x) e^{-ikx} and i k f(x) e^{-ikx} over [-10, 10] with complex balls (Arb) and widens
#   both by the tail bound e^{-90}; the two enclosures must overlap
SEED = 0
RI = RealIntervalField(128)
CI = ComplexIntervalField(128)
CB = ComplexBallField(128)
def enclose(g, k):
    val = CB.integral(lambda x, _: g(x) * (-CB(I) * CB(k) * x).exp(), -10, 10)
    t = RI(exp(-90))
    return CI(RI(val.real().lower() - t.upper(), val.real().upper() + t.upper()),
              RI(val.imag().lower() - t.upper(), val.imag().upper() + t.upper()))
gauss = lambda x: (-x * x).exp()
gauss_d = lambda x: -2 * x * (-x * x).exp()
xgauss = lambda x: x * (-x * x).exp()
xgauss_d = lambda x: (1 - 2 * x * x) * (-x * x).exp()
instances = []
CASES = [(['fun x => Complex.exp (-(x : ℂ) ^ 2)', '_', '_', '_', '0'], True, (gauss, gauss_d, 0)),
         (['fun x => Complex.exp (-(x : ℂ) ^ 2)', '_', '_', '_', '1'], False, (gauss, gauss_d, 1)),
         (['fun x => (x : ℂ) * Complex.exp (-(x : ℂ) ^ 2)', '_', '_', '_', '2'], False, (xgauss, xgauss_d, 2))]
for values, edge, (f, fp, k) in CASES:
    L = enclose(fp, k)
    R = CI(I * k) * enclose(f, k)
    d = CI(L) - CI(R)
    L, R = RI(d.real().abs() + d.imag().abs()), RI(0)
    instances.append(oracle_instance(*values, lhs=L, rhs=R, relation='==', edge=edge))
    print(values, L, R)
