# name: cm-central-potential-grad-test
# technique: test a claim no recipe certifies, at instances Sage evaluates exactly (test_fallback recipe, oracle
#   instances; the binders are functions)
# goal: sage_test KaluzaKlein.ClassicalMechanics.CentralForces.central_potential_grad file "cm-central-potential-grad-test.sage"
# description: ∂_x V(|x|) = -f(|x|) x/|x| for V' = -f, at (V, f) = (0, 0) at (1, 0, 0) (edge), (r²/2, -r) at (1, 2, 2),
#   (-r³/3, r²) at (3, 4, 0)
SEED = 0  # the instances are listed by hand
X, Y, Z, r = SR.var('X Y Z r')
def inst(Vl, fl, V, f, x, y, z, edge):
    rr = sqrt(X**2 + Y**2 + Z**2)
    lhs = diff(SR(V).subs(r=rr), X).subs({X: x, Y: y, Z: z})
    R = sqrt(x**2 + y**2 + z**2)
    rhs = -(SR(f).subs(r=R) * x / R)
    return oracle_instance(Vl, fl, '_', str(x), str(y), str(z), '_', lhs=lhs, rhs=rhs, relation='==', edge=edge)
instances = [inst('fun _ : ℝ => (0 : ℝ)', 'fun _ : ℝ => (0 : ℝ)', 0, 0, 1, 0, 0, True),
             inst('fun r : ℝ => r ^ 2 / 2', 'fun r : ℝ => -r', r**2 / 2, -r, 1, 2, 2, False),
             inst('fun r : ℝ => -r ^ 3 / 3', 'fun r : ℝ => r ^ 2', -r**3 / 3, r**2, 3, 4, 0, False)]
