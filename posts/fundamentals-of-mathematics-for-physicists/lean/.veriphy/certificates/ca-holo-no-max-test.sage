# name: ca-holo-no-max-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_strict_local_max
# description: Re z, Im z, Re z^2 have no strict local maximum at 0, 1/2, i; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun z : ℂ => z.re)', 'Set.univ', 'isOpen_univ', '(fun _ _ => analyticAt_id.harmonicAt_re)', '0', '(Set.mem_univ _)'], 'exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_strict_local_max_aux (fun z : ℂ => z.re) Set.univ isOpen_univ (fun _ _ => analyticAt_id.harmonicAt_re) 0 (Set.mem_univ _)', True),
    (['(fun z : ℂ => z.im)', '(Metric.ball 0 1)', 'Metric.isOpen_ball', '(fun _ _ => analyticAt_id.harmonicAt_im)', '(1 / 2)', '(by simp; norm_num)'], 'exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_strict_local_max_aux (fun z : ℂ => z.im) (Metric.ball 0 1) Metric.isOpen_ball (fun _ _ => analyticAt_id.harmonicAt_im) (1 / 2) (by simp; norm_num)', False),
    (['(fun z : ℂ => (z ^ 2).re)', 'Set.univ', 'isOpen_univ', '(fun _ _ => (analyticAt_id.pow 2).harmonicAt_re)', 'Complex.I', '(Set.mem_univ _)'], 'exact KaluzaKlein.ComplexAnalysis.HarmonicFunctions.no_strict_local_max_aux (fun z : ℂ => (z ^ 2).re) Set.univ isOpen_univ (fun _ _ => (analyticAt_id.pow 2).harmonicAt_re) Complex.I (Set.mem_univ _)', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
