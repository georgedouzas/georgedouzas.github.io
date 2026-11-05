# name: ra-spherical-pole-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Calculus.spherical_pole file "ra-spherical-pole-test.sage"
# description: at θ = 0 the spherical map is (0, 0, r) for every φ
SEED = 0  # the instances are listed, no sampling
PROOF = 'simp'
CASES = [(['1', '0'], True, None), (['2', '1'], False, None), (['3', 'Real.pi'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
