# name: ca-num-topology-test
# technique: test fallback (test_fallback recipe): the claim at instances the kernel checks by computation
# goal: sage_test KaluzaKlein.ComplexAnalysis.Modulus.topology_as_R2 file "ca-num-topology-test.sage"
# description: no recipe certifies the general statement; instances with an edge case first
SEED = 0
RI = RealIntervalField(128)
instances = []
instances.append(kernel_instance(*['∅'], proof="let e := Complex.equivRealProdCLM.toHomeomorph\nhave he : ∀ t : Set ℂ, e '' t = Complex.equivRealProdCLM '' t := fun t => rfl\nrefine ⟨?_, ?_, ?_, ?_⟩\n· rw [← he, e.isOpen_image]\n· rw [← he, e.isClosed_image]\n· rw [← he, e.isCompact_image]\n· rw [← he, e.isConnected_image]", edge=True))
instances.append(kernel_instance(*['Set.univ'], proof="let e := Complex.equivRealProdCLM.toHomeomorph\nhave he : ∀ t : Set ℂ, e '' t = Complex.equivRealProdCLM '' t := fun t => rfl\nrefine ⟨?_, ?_, ?_, ?_⟩\n· rw [← he, e.isOpen_image]\n· rw [← he, e.isClosed_image]\n· rw [← he, e.isCompact_image]\n· rw [← he, e.isConnected_image]", edge=False))
instances.append(kernel_instance(*['Metric.ball 0 1'], proof="let e := Complex.equivRealProdCLM.toHomeomorph\nhave he : ∀ t : Set ℂ, e '' t = Complex.equivRealProdCLM '' t := fun t => rfl\nrefine ⟨?_, ?_, ?_, ?_⟩\n· rw [← he, e.isOpen_image]\n· rw [← he, e.isClosed_image]\n· rw [← he, e.isCompact_image]\n· rw [← he, e.isConnected_image]", edge=False))
print([(x.values, x.kind, x.verdict) for x in instances])
