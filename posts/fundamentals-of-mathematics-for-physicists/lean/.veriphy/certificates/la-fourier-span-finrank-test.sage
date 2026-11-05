# name: la-fourier-span-finrank-test
# technique: test a claim no recipe certifies, at instances the kernel checks by computation (test_fallback)
# goal: sage_test KaluzaKlein.LinearAlgebra.Dimension.fourier_span_finrank_ne file "la-fourier-span-finrank-test.sage"
# description: the span of all the modes has no finite dimension N, at N = 0, 1, 2, 5: it contains 2N + 3 independent modes.
SEED = 0  # EDIT: the sampler's seed, kept in the test record
PROOF = None  # EDIT: the tactic block proving the claim at one instance
CASES = [  # EDIT: the instances as ([values as Lean text], edge?[, proof]) tuples
    (['0'], True, 'rintro ⟨h, hN⟩\nhave hle : Submodule.span ℝ (Set.range (fourierModes ∘ Sum.map (fun k : Fin 2 => (k : ℕ)) (fun k : Fin 1 => (k : ℕ)))) ≤ Submodule.span ℝ (Set.range fourierModes) :=\n  Submodule.span_mono (Set.range_comp_subset_range _ _)\nhave := Submodule.finrank_mono hle\nrw [finrank_span_fourier_modes, hN] at this\nomega'),
    (['1'], False, 'rintro ⟨h, hN⟩\nhave hle : Submodule.span ℝ (Set.range (fourierModes ∘ Sum.map (fun k : Fin 3 => (k : ℕ)) (fun k : Fin 2 => (k : ℕ)))) ≤ Submodule.span ℝ (Set.range fourierModes) :=\n  Submodule.span_mono (Set.range_comp_subset_range _ _)\nhave := Submodule.finrank_mono hle\nrw [finrank_span_fourier_modes, hN] at this\nomega'),
    (['2'], False, 'rintro ⟨h, hN⟩\nhave hle : Submodule.span ℝ (Set.range (fourierModes ∘ Sum.map (fun k : Fin 4 => (k : ℕ)) (fun k : Fin 3 => (k : ℕ)))) ≤ Submodule.span ℝ (Set.range fourierModes) :=\n  Submodule.span_mono (Set.range_comp_subset_range _ _)\nhave := Submodule.finrank_mono hle\nrw [finrank_span_fourier_modes, hN] at this\nomega'),
    (['5'], False, 'rintro ⟨h, hN⟩\nhave hle : Submodule.span ℝ (Set.range (fourierModes ∘ Sum.map (fun k : Fin 7 => (k : ℕ)) (fun k : Fin 6 => (k : ℕ)))) ≤ Submodule.span ℝ (Set.range fourierModes) :=\n  Submodule.span_mono (Set.range_comp_subset_range _ _)\nhave := Submodule.finrank_mono hle\nrw [finrank_span_fourier_modes, hN] at this\nomega'),
]
cases = CASES
instances = [kernel_instance(*c[0], proof=c[2] if len(c) > 2 else PROOF, edge=c[1]) for c in cases]
