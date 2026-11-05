# name: cm-bertrand-test
# technique: test a claim no recipe certifies, at instances Sage evaluates (test_fallback recipe, oracle instances;
#   the claim quantifies over all bounded orbits of a force law)
# goal: sage_test KaluzaKlein.ClassicalMechanics.CentralForces.bertrand_theorem file "cm-bertrand-test.sage"
# description: the bounded orbits of f = -k r^α all close iff α = -2 or α = 1, at α = -2 (edge, Kepler), 1 (oscillator),
#   -1 (logarithmic potential): the orbit from (x, y, ẋ, ẏ) = (1, 0, 0, 4/5) with k = 1 is integrated numerically
#   (relative tolerance 1e-11) and closes when the state returns to its initial value (distance < 1e-6) before t = 60
SEED = 0  # the instances are listed by hand
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
def closes(alpha):
    def rhs(t, s):
        rr = np.hypot(s[0], s[1])
        return [s[2], s[3], -rr**(alpha - 1) * s[0], -rr**(alpha - 1) * s[1]]
    s0 = np.array([1.0, 0.0, 0.0, 0.8])
    sol = solve_ivp(rhs, [0, 60], s0, rtol=1e-11, atol=1e-13, dense_output=True)
    dist = lambda t: float(np.linalg.norm(sol.sol(t) - s0))
    ts = np.linspace(1, 60, 59001)
    ds = np.array([dist(t) for t in ts])
    i = int(np.argmin(ds))
    best = minimize_scalar(dist, bounds=(ts[max(i - 2, 0)], ts[min(i + 2, len(ts) - 1)]), method='bounded')
    return best.fun < 1e-6
def inst(alpha, edge):
    return oracle_instance(str(alpha), lhs=B(closes(float(alpha))), rhs=B(alpha in (-2, 1)), relation='==', edge=edge)
instances = [inst(-2, True), inst(1, False), inst(-1, False)]
