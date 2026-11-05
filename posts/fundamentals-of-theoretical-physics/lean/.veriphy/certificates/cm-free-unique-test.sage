# name: cm-free-unique-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly (test_fallback recipe, oracle
#   instances; the binder is a function)
# goal: sage_test KaluzaKlein.ClassicalMechanics.Momentum.free_unique file "cm-free-unique-test.sage"
# description: x t = x 0 + x' 0 t for motions with zero acceleration, at x = 0 (edge), 1 + 2t at t = 2, 5 - t at t = -3
SEED = 0  # the instances are listed by hand
t = SR.var('t')
body_lean = {'0': '(0 : ℝ)', '1 + 2*t': '1 + 2 * t', '5 - t': '5 - t'}
def inst(body, t0, edge):
    x = SR(body)
    return oracle_instance(f'fun t : ℝ => {body_lean[body]}', '_', '_', '_', str(t0),
                           lhs=x.subs(t=t0), rhs=x.subs(t=0) + diff(x, t).subs(t=0) * t0, relation='==', edge=edge)
instances = [inst('0', 1, True), inst('1 + 2*t', 2, False), inst('5 - t', -3, False)]
