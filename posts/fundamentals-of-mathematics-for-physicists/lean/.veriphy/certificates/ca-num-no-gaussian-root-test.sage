# name: ca-num-no-gaussian-root-test
# technique: test fallback (test_fallback recipe): the claim at instances Sage evaluates exactly or in interval arithmetic
# goal: sage_test KaluzaKlein.ComplexAnalysis.NumbersWorked.no_root_gaussian file "ca-num-no-gaussian-root-test.sage"
# description: Sage evaluates both quadratics at a + bi exactly in ℚ(i) (the product of their squared moduli is positive); instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
CI = ComplexIntervalField(128)
def cdiff(a, b):
    d = CI(a) - CI(b)
    return RI(d.real().abs() + d.imag().abs())
CB = ComplexBallField(128)
def bdiff(a, b):
    d = a - b
    return RI(d.real().abs()) + RI(d.imag().abs())
instances.append(oracle_instance(*['0', '0'], lhs=RI(AA(abs(QQbar(0+(0)*I)^2+QQbar(0+(0)*I)+1)^2*abs(QQbar(0+(0)*I)^2-QQbar(0+(0)*I)+1)^2)), rhs=RI(AA(0)), relation='>', edge=True))
instances.append(oracle_instance(*['1', '2'], lhs=RI(AA(abs(QQbar(1+(2)*I)^2+QQbar(1+(2)*I)+1)^2*abs(QQbar(1+(2)*I)^2-QQbar(1+(2)*I)+1)^2)), rhs=RI(AA(0)), relation='>', edge=False))
instances.append(oracle_instance(*['(-1 / 2)', '(1 / 2)'], lhs=RI(AA(abs(QQbar(-1/2+(1/2)*I)^2+QQbar(-1/2+(1/2)*I)+1)^2*abs(QQbar(-1/2+(1/2)*I)^2-QQbar(-1/2+(1/2)*I)+1)^2)), rhs=RI(AA(0)), relation='>', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
