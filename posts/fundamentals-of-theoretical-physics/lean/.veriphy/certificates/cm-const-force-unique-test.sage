# name: cm-const-force-unique-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly (test_fallback recipe, oracle
#   instances; the binder is a function)
# goal: sage_test KaluzaKlein.ClassicalMechanics.Momentum.const_force_unique file "cm-const-force-unique-test.sage"
# description: x t = x 0 + x' 0 t + a t²/2 for motions with constant acceleration, at x = 0 (edge), 1 + 2t + 3t²/2 at
#   t = 2, 5 - t - t²/2 at t = -3
SEED = 0  # the instances are listed by hand
t = SR.var('t')
def inst(body, a, t0, edge):
    x = SR(body)
    return oracle_instance(f'fun t : ℝ => {body_lean[body]}', str(a), '_', '_', '_', str(t0),
                           lhs=x.subs(t=t0), rhs=x.subs(t=0) + diff(x, t).subs(t=0) * t0 + a * t0**2 / 2,
                           relation='==', edge=edge)
body_lean = {'0': '(0 : ℝ)', '1 + 2*t + 3*t^2/2': '1 + 2 * t + 3 * t ^ 2 / 2', '5 - t - t^2/2': '5 - t - t ^ 2 / 2'}
instances = [inst('0', 0, 1, True), inst('1 + 2*t + 3*t^2/2', 3, 2, False), inst('5 - t - t^2/2', -1, -3, False)]
