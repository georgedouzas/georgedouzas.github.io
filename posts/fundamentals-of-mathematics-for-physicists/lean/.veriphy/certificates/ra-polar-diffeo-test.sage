# name: ra-polar-diffeo-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.polar_diffeo file "ra-polar-diffeo-test.sage"
# description: the target of polarCoord and the collapse of r = 0 at sample angles
SEED = 0  # the instances are listed, no sampling
PROOF = 'exact ⟨rfl, by simp⟩'
CASES = [(['0'], True, None), (['1'], False, None), (['Real.pi / 2'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
