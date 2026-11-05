# name: cm-drag-momentum-changes-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback recipe)
# goal: sage_test KaluzaKlein.ClassicalMechanics.Momentum.drag_momentum_changes file "cm-drag-momentum-changes-test.sage"
# description: -γ v₀ e^{-γ t/m} ≠ 0 at (γ, v₀, m, t) = (1, 1, 1, 0) (edge), (1/2, 3, 2, 5), (-1, 1/10, 1, -2)
SEED = 0  # the instances are listed by hand
PROOF = 'norm_num [Real.exp_ne_zero]'
CASES = [(['1', '1', '(by norm_num)', '(by norm_num)', '1', '0'], True),
         (['1/2', '3', '(by norm_num)', '(by norm_num)', '2', '5'], False),
         (['-1', '1/10', '(by norm_num)', '(by norm_num)', '1', '-2'], False)]
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in CASES]
