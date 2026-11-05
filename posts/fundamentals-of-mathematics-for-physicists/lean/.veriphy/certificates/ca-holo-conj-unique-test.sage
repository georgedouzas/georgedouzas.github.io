# name: ca-holo-conj-unique-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.HarmonicFunctions.conjugate_unique
# description: conjugates Im z, Im z + c of Re z differ by constants; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['(fun z : ℂ => z.re)', '(fun z : ℂ => z.im)', '(fun z : ℂ => z.im)', 'Set.univ', 'isOpen_univ', 'isPreconnected_univ', '(by simp only [Complex.re_add_im]; fun_prop)', '(by simp only [Complex.re_add_im]; fun_prop)', '0', '(Set.mem_univ _)', 'Complex.I', '(Set.mem_univ _)'], 'ring', True),
    (['(fun z : ℂ => z.re)', '(fun z : ℂ => z.im)', '(fun z : ℂ => z.im + 1)', 'Set.univ', 'isOpen_univ', 'isPreconnected_univ', '(by simp only [Complex.re_add_im]; fun_prop)', '(by simp only [Complex.ofReal_add, Complex.ofReal_one, add_mul, ← add_assoc, Complex.re_add_im, one_mul]; fun_prop)', '1', '(Set.mem_univ _)', '(2 * Complex.I)', '(Set.mem_univ _)'], 'ring', False),
    (['(fun z : ℂ => z.re)', '(fun z : ℂ => z.im)', '(fun z : ℂ => z.im - 2)', '(Metric.ball 0 1)', 'Metric.isOpen_ball', '(convex_ball (0 : ℂ) 1).isPreconnected', '(by simp only [Complex.re_add_im]; fun_prop)', '(by simp only [Complex.ofReal_sub, Complex.ofReal_ofNat, sub_mul, ← add_sub_assoc, Complex.re_add_im]; fun_prop)', '0', '(Metric.mem_ball_self one_pos)', '(Complex.I / 2)', '(by simp; norm_num)'], 'ring', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
