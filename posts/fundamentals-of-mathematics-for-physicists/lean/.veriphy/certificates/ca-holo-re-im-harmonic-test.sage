# name: ca-holo-re-im-harmonic-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HarmonicFunctions.re_im_harmonic
# description: Re and Im of w, w^2, w^3 + w are harmonic; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun w : ℂ => w)', 'Set.univ', 'isOpen_univ', '(by fun_prop)', '0', '(Set.mem_univ _)'], 'have h : AnalyticAt ℂ (fun w : ℂ => w) (0) := DifferentiableOn.analyticAt (s := Set.univ) (by fun_prop) (Filter.univ_mem)\nexact ⟨h.harmonicAt_re, h.harmonicAt_im⟩', True),
    (['(fun w : ℂ => w ^ 2)', 'Set.univ', 'isOpen_univ', '(by fun_prop)', '(1 + Complex.I)', '(Set.mem_univ _)'], 'have h : AnalyticAt ℂ (fun w : ℂ => w ^ 2) (1 + Complex.I) := DifferentiableOn.analyticAt (s := Set.univ) (by fun_prop) (Filter.univ_mem)\nexact ⟨h.harmonicAt_re, h.harmonicAt_im⟩', False),
    (['(fun w : ℂ => w ^ 3 + w)', '(Metric.ball 0 1)', 'Metric.isOpen_ball', '(by fun_prop)', '0', '(Metric.mem_ball_self one_pos)'], 'have h : AnalyticAt ℂ (fun w : ℂ => w ^ 3 + w) (0) := DifferentiableOn.analyticAt (s := Metric.ball 0 1) (by fun_prop) (Metric.isOpen_ball.mem_nhds (Metric.mem_ball_self one_pos))\nexact ⟨h.harmonicAt_re, h.harmonicAt_im⟩', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
