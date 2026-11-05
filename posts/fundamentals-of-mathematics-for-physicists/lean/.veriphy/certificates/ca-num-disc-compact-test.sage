# name: ca-num-disc-compact-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Modulus.closed_disc_compact file "ca-num-disc-compact-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['0', '0'], proof='exact isCompact_closedBall _ _', edge=True))
instances.append(kernel_instance(*['0', '1'], proof='exact isCompact_closedBall _ _', edge=False))
instances.append(kernel_instance(*['Complex.I', '2'], proof='exact isCompact_closedBall _ _', edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
