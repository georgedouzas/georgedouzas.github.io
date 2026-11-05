# name: ra-exp-term-deriv-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Series.exp_term_deriv file "ra-exp-term-deriv-test.sage"
# description: (x^(k+1)/(k+1)!)' = x^k/k! at sample k, x
SEED = 0  # the instances are listed, no sampling
PROOF = None
CASES = [(['0', '0'], True, 'have h := (hasDerivAt_pow (0 + 1) (0 : ℝ)).div_const (((0 + 1 : ℕ)) ! : ℝ)\nconvert h using 1\nnorm_num [Nat.factorial]'), (['2', '3'], False, 'have h := (hasDerivAt_pow (2 + 1) (3 : ℝ)).div_const (((2 + 1 : ℕ)) ! : ℝ)\nconvert h using 1\nnorm_num [Nat.factorial]'), (['4', '-1'], False, 'have h := (hasDerivAt_pow (4 + 1) (-1 : ℝ)).div_const (((4 + 1 : ℕ)) ! : ℝ)\nconvert h using 1\nnorm_num [Nat.factorial]')]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
