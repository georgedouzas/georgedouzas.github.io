# name: ra-smooth-rules-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.smooth_rules file "ra-smooth-rules-test.sage"
# description: the rules at explicit smooth functions
SEED = 0  # the instances are listed, no sampling
PROOF = 'refine ⟨by fun_prop, by fun_prop, ?_⟩\nexact ContDiff.comp (by fun_prop) (by fun_prop)'
CASES = [(['Real.sin', 'Real.exp', 'Real.contDiff_sin', 'Real.contDiff_exp'], False, None), (['(fun x : ℝ => x ^ 2)', 'Real.cos', '(contDiff_id.pow 2)', 'Real.contDiff_cos'], False, None), (['(fun _ : ℝ => (0 : ℝ))', '(fun _ : ℝ => (1 : ℝ))', 'contDiff_const', 'contDiff_const'], True, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
