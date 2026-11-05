# name: ca-holo-mean-value-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HarmonicFunctions.harmonic_mean_value
# description: the mean value property for Re z, Im z, Re z^2; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun z : ℂ => z.re)', '0', '0', '(fun _ _ => analyticAt_id.harmonicAt_re)'], 'exact InnerProductSpace.HarmonicOnNhd.circleAverage_eq (fun _ _ => analyticAt_id.harmonicAt_re)', True),
    (['(fun z : ℂ => z.im)', '1', '2', '(fun _ _ => analyticAt_id.harmonicAt_im)'], 'exact InnerProductSpace.HarmonicOnNhd.circleAverage_eq (fun _ _ => analyticAt_id.harmonicAt_im)', False),
    (['(fun z : ℂ => (z ^ 2).re)', 'Complex.I', '1', '(fun _ _ => (analyticAt_id.pow 2).harmonicAt_re)'], 'exact InnerProductSpace.HarmonicOnNhd.circleAverage_eq (fun _ _ => (analyticAt_id.pow 2).harmonicAt_re)', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
