# name: ra-gamma-props-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.MultipleIntegrals.gamma_props file "ra-gamma-props-test.sage"
# description: convergence, Γ(s + 1) = sΓ(s) and Γ(n + 1) = n! at sample s, n (mathlib's Γ)
SEED = 0  # the instances are listed, no sampling
PROOF = 'exact ⟨Real.GammaIntegral_convergent (by norm_num), Real.Gamma_add_one (by norm_num), Real.Gamma_nat_eq_factorial _⟩'
CASES = [(['1', '(by norm_num)', '0'], True, None), (['1 / 2', '(by norm_num)', '3'], False, None), (['5', '(by norm_num)', '4'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
