# name: cm-drag-not-conservative-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback recipe)
# goal: sage_test KaluzaKlein.ClassicalMechanics.Energy.drag_not_conservative file "cm-drag-not-conservative-test.sage"
# description: -γ v is no gradient -V'(x), at γ = 1 (edge), -2, 1/3: the values at v = 0 and v = 1 differ
SEED = 0  # the instances are listed by hand
PROOF = 'rintro ⟨V, h⟩\nhave h0 := h 0 0\nhave h1 := h 0 1\nnorm_num at h0 h1\nlinarith'
CASES = [(['1', '(by norm_num)'], True), (['-2', '(by norm_num)'], False), (['1/3', '(by norm_num)'], False)]
instances = [kernel_instance(*values, proof=PROOF, edge=edge) for values, edge in CASES]
