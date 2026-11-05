# name: ca-holo-cayley-im-test
# technique: test fallback (kernel-checked instances)
# goal: sage_test KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_inv_im
# description: Im z(w) > 0 at w = 0, 1/2, i/2; each instance proved by computation or mathlib, independently of the claim.
SEED = 0
CASES = [
    (['0', '(by simp)'], 'exact KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_inv_im_aux 0 (by simp)', True),
    (['(1 / 2)', '(by norm_num)'], 'exact KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_inv_im_aux (1 / 2) (by norm_num)', False),
    (['(Complex.I / 2)', '(by simp; norm_num)'], 'exact KaluzaKlein.ComplexAnalysis.ConformalMaps.cayley_inv_im_aux (Complex.I / 2) (by simp; norm_num)', False),
]
instances = [kernel_instance(*values, proof=proof, edge=edge) for values, proof, edge in CASES]
