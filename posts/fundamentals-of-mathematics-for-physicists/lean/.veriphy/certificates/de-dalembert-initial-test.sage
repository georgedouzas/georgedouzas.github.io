# name: de-dalembert-initial-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly or in interval arithmetic
#   (test_fallback recipe, oracle instances; the binders are functions, matrices or need exact symbolic values)
# goal: sage_test KaluzaKlein.DifferentialEquations.WaveEquation.dalembert_initial file "de-dalembert-initial-test.sage"
# description: d'Alembert's u has u(0, x) = φ and u_t(0, x) = ψ by Sage for generic φ, Ψ (symbolic functions), and c = 1 with φ = Ψ = 0 (edge)
SEED = 0  # the instances are listed by hand
x = SR.var('x')
RIF = RealIntervalField(64)
def B(c):
    return ZZ(1) if bool(c) else ZZ(0)
t_, c_ = var('t_ c_'); ph = function('ph'); Ps = function('Ps')
instances = [
    oracle_instance('_', '_', '_', '_', '_', '_', 'c_', 'x', '_',
                    lhs=((ph(x - c_*t_) + ph(x + c_*t_))/2 + (Ps(x + c_*t_) - Ps(x - c_*t_))/(2*c_)).subs(t_=0),
                    rhs=ph(x), relation='==', edge=False),
    oracle_instance('_', '_', '_', '_', '_', '_', 'c_', 'x', '_',
                    lhs=diff((ph(x - c_*t_) + ph(x + c_*t_))/2 + (Ps(x + c_*t_) - Ps(x - c_*t_))/(2*c_), t_).subs(t_=0),
                    rhs=diff(Ps(x), x), relation='==', edge=False),
    oracle_instance('fun _ => 0', 'fun _ => 0', 'fun _ => 0', 'fun _ => 0', '_', '_', '1', '0', '_',
                    lhs=0,
                    rhs=0, relation='==', edge=True),
]
