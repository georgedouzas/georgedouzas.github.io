# name: de-nonlinear-sin
# technique: a failure of map_smul at a point where Sage evaluates sin exactly (nonlinearity recipe, transcendental map)
# goal: ¬ ∃ f : ℝ →ₗ[ℝ] ℝ, ∀ x, f x = Real.sin x
# description: Sage finds that homogeneity fails at (2, π/2): sin π = 0 but 2 sin(π/2) = 2; Lean specialises
#   f.map_smul there and evaluates with the special values of sin.
a, b = 2, pi / 2
assert sin(a * b) != a * sin(b), 'homogeneity holds at the chosen point'
proof = tactic('rintro ⟨f, hf⟩\nhave h := f.map_smul (2 : ℝ) (Real.pi / 2)\nrw [hf, hf, smul_eq_mul, smul_eq_mul,\n'
               '  show (2 : ℝ) * (Real.pi / 2) = Real.pi by ring] at h\nnorm_num at h')
