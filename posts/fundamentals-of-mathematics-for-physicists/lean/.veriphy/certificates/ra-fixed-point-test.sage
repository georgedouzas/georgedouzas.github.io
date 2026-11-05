# name: ra-fixed-point-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.RealAnalysis.Continuity.fixed_point file "ra-fixed-point-test.sage"
# description: a fixed point Sage solves for (1 - x: 1/2; x²: 0; (x + 1)/2: 1)
SEED = 0  # the instances are listed, no sampling
PROOF = None
CASES = [(['(fun x : ℝ => 1 - x)', '(by fun_prop)', '(fun x hx => ⟨by simp only [Set.mem_Icc] at hx; linarith [hx.2], by simp only [Set.mem_Icc] at hx; linarith [hx.1]⟩)'], False, 'exact ⟨1 / 2, by norm_num, by norm_num⟩'), (['(fun x : ℝ => x ^ 2)', '(by fun_prop)', '(fun x hx => ⟨by positivity, by simp only [Set.mem_Icc] at hx; nlinarith [hx.1, hx.2]⟩)'], True, 'exact ⟨0, by norm_num, by norm_num⟩'), (['(fun x : ℝ => (x + 1) / 2)', '(by fun_prop)', '(fun x hx => ⟨by simp only [Set.mem_Icc] at hx; linarith [hx.1], by simp only [Set.mem_Icc] at hx; linarith [hx.2]⟩)'], False, 'exact ⟨1, by norm_num, by norm_num⟩')]
instances = [kernel_instance(*values, proof=(p if p is not None else PROOF), edge=edge) for values, edge, p in CASES]
