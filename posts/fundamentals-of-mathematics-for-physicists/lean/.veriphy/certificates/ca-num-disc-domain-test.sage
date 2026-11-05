# name: ca-num-disc-domain-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Modulus.disc_domain file "ca-num-disc-domain-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0', '1', 'one_pos'], proof='exact ⟨Metric.isOpen_ball, (convex_ball _ _).isConnected (Metric.nonempty_ball.mpr (by norm_num))⟩', edge=True))
instances.append(kernel_instance(*['Complex.I', '2', 'two_pos'], proof='exact ⟨Metric.isOpen_ball, (convex_ball _ _).isConnected (Metric.nonempty_ball.mpr (by norm_num))⟩', edge=False))
instances.append(kernel_instance(*['1', '(1 / 2)', '(by norm_num)'], proof='exact ⟨Metric.isOpen_ball, (convex_ball _ _).isConnected (Metric.nonempty_ball.mpr (by norm_num))⟩', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
