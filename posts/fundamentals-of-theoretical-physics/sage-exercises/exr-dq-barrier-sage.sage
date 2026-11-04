# Exercise exr-dq-barrier-sage: the Dirac square barrier V on |x| < a (m = 1, E = 3, V = 8) by matching at x = -a and x = a; T against
# 4 kappa^2/(4 kappa^2 + (1 - kappa^2)^2 sin^2(2 q a)), and the first resonance 2|q|a = pi.
m, E, V = 1, 3, 8
k = sqrt(E^2 - m^2); q = -sqrt((V - E)^2 - m^2)          # group velocity to the right inside the barrier
sp = lambda p, eps: vector(CDF, [1, p / (eps + m)])    # spinor of momentum p and kinetic energy eps
def T_num(a):
    # unknowns B (reflected), C, D (inside, e^{iqx}, e^{-iqx}), F (transmitted)
    M = matrix(CDF, 4, 4); rhs = vector(CDF, 4)
    xl, xr = -a, a
    inc = sp(k, E) * exp(I * k * xl)
    cols = [-sp(-k, E) * exp(-I * k * xl), sp(q, E - V) * exp(I * q * xl), sp(-q, E - V) * exp(-I * q * xl), vector(CDF, [0, 0])]
    cols2 = [vector(CDF, [0, 0]), sp(q, E - V) * exp(I * q * xr), sp(-q, E - V) * exp(-I * q * xr), -sp(k, E) * exp(I * k * xr)]
    for j in range(4):
        M[0, j], M[1, j] = cols[j][0], cols[j][1]
        M[2, j], M[3, j] = cols2[j][0], cols2[j][1]
    rhs[0], rhs[1] = inc[0], inc[1]
    B, C, D, F = M.solve_right(rhs)
    return abs(F)^2
kap = (q / k * (E + m) / (E - V + m)).n()
T_cf = lambda a: 4 * kap^2 / (4 * kap^2 + (1 - kap^2)^2 * sin(2 * abs(q.n()) * a)^2)
for a in (0.1, 0.2, 0.25, 0.5):
    print("a = %.2f: T (matching) = %.6f, T (formula) = %.6f" % (a, T_num(a), T_cf(a)))
ares = (pi / (2 * abs(q))).n()
print("first resonance at a = pi/(2 sqrt 24) = %.4f: T = %.10f" % (ares, T_num(ares)))
