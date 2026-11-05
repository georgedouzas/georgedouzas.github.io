# name: ra-gamma-factorial-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.gamma_factorial file "ra-gamma-factorial-test.sage"
# description: Γ(n + 1) = n! (mathlib's Γ at n = 0, 1, 4)
SEED = 0  # the instances are listed, no sampling
PROOF = 'rw [Real.Gamma_nat_eq_factorial]'
CASES = [(['0'], True, None), (['1'], False, None), (['4'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
