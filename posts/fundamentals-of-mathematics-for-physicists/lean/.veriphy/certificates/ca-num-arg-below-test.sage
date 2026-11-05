# name: ca-num-arg-below-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Argument.arg_tendsto_below file "ca-num-arg-below-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['(-1)', '(by norm_num)', '(by simp)'], proof='exact Complex.tendsto_arg_nhdsWithin_im_neg_of_re_neg_of_im_zero (by norm_num) (by simp)', edge=True))
instances.append(kernel_instance(*['(-4)', '(by norm_num)', '(by simp)'], proof='exact Complex.tendsto_arg_nhdsWithin_im_neg_of_re_neg_of_im_zero (by norm_num) (by simp)', edge=False))
instances.append(kernel_instance(*['(-1 / 2)', '(by norm_num)', '(by simp)'], proof='exact Complex.tendsto_arg_nhdsWithin_im_neg_of_re_neg_of_im_zero (by norm_num) (by simp)', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
