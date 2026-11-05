# name: ra-sinf-ioi-unbounded-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Sequences.sInf_Ioi_unbounded file "ra-sinf-ioi-unbounded-test.sage"
# description: inf (a, ∞) = a and (a, ∞) unbounded, at a = 0 and others
SEED = 0  # the instances are listed, no sampling
PROOF = 'exact ⟨isGLB_Ioi, csInf_Ioi, not_bddAbove_Ioi _⟩'
CASES = [(['0'], True, None), (['1'], False, None), (['-5 / 2'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
