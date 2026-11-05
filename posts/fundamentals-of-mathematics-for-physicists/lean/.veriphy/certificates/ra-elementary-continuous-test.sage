# name: ra-elementary-continuous-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.elementary_continuous file "ra-elementary-continuous-test.sage"
# description: continuity of polynomials (and exp, sin, cos, log) at sample polynomials
SEED = 0  # the instances are listed, no sampling
PROOF = 'refine ⟨Polynomial.continuous _, Real.continuous_exp, Real.continuous_sin, Real.continuous_cos, Real.continuousOn_log.mono (fun x hx => ne_of_gt hx)⟩'
CASES = [(['(0 : Polynomial ℝ)'], True, None), (['(Polynomial.X ^ 2 + 1 : Polynomial ℝ)'], False, None), (['(Polynomial.C 3 * Polynomial.X ^ 5 : Polynomial ℝ)'], False, None)]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
