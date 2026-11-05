# name: cm-gradient-no-work-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly (test_fallback recipe, oracle
#   instances; the binder is a function)
# goal: sage_test KaluzaKlein.ClassicalMechanics.Energy.gradient_no_work file "cm-gradient-no-work-test.sage"
# description: ∫₀^{2π} g' = 0 for g with g(2π) = g(0), at g = 0 (edge), cos² s, sin s cos s
SEED = 0  # the instances are listed by hand
s = SR.var('s')
def inst(lean, g, edge):
    return oracle_instance(lean, '_', '_', '_', lhs=integrate(diff(SR(g), s), s, 0, 2 * pi), rhs=0, relation='==', edge=edge)
instances = [inst('fun _ : ℝ => (0 : ℝ)', 0, True), inst('fun s : ℝ => Real.cos s ^ 2', cos(s)**2, False),
             inst('fun s : ℝ => Real.sin s * Real.cos s', sin(s) * cos(s), False)]
