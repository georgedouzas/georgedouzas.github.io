# name: cm-particles-dof-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback recipe)
# goal: sage_test KaluzaKlein.ClassicalMechanics.ConfigurationSpace.particles_dof file "cm-particles-dof-test.sage"
# description: dim ℝ^{3N} = 3N at N = 0 (edge), 1, 2, 5
SEED = 0  # the instances are listed by hand
PROOF = 'simp [Module.finrank_fin_fun]'
CASES = [(['0'], True), (['1'], False), (['2'], False), (['5'], False)]
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in CASES]
