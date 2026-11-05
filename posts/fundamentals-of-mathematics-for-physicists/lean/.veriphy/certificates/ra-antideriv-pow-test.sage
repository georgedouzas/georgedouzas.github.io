# name: ra-antideriv-pow-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Integral.antideriv_pow file "ra-antideriv-pow-test.sage"
# description: (x^(n+1)/(n+1))' = xⁿ at sample n, x
SEED = 0  # the instances are listed, no sampling
PROOF = None
CASES = [(['0', '3'], True, 'have h := (hasDerivAt_pow (0 + 1) (3 : ℝ)).div_const (((0 : ℕ) : ℝ) + 1)\nconvert h using 1\nnorm_num'), (['2', '1'], False, 'have h := (hasDerivAt_pow (2 + 1) (1 : ℝ)).div_const (((2 : ℕ) : ℝ) + 1)\nconvert h using 1\nnorm_num'), (['4', '-2'], False, 'have h := (hasDerivAt_pow (4 + 1) (-2 : ℝ)).div_const (((4 : ℕ) : ℝ) + 1)\nconvert h using 1\nnorm_num')]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
