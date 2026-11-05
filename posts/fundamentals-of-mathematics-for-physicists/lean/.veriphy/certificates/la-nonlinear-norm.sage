# name: la-nonlinear-norm
# technique: a failure of homogeneity at points Sage finds (nonlinearity recipe, adapted to a bilinear map that
#   vanishes at zero: the template certifies only f 0 0 ≠ 0, so the failing law is map_smul in the first argument)
# goal: ¬ ∃ B : LinearMap.BilinForm ℝ (EuclideanSpace ℝ (Fin 2)), ∀ v w, B v w = ‖v‖ * ‖w‖
# description: Sage tests g(c v, w) = c g(v, w) for g(v, w) = |v| |w| at v = w = e₁ and small scalars c, and finds
#   c = -1; Lean specialises B at (-e₁, e₁), rewrites by linearity and the norm, and computes 1 ≠ -1.
g = lambda c: abs(c) * 1 * 1  # EDIT: g(c e₁, e₁) for the unit vector e₁
c = next(c for c in (2, -1, QQ(1) / 2, 3, -2) if g(c) != c * g(1))
print('homogeneity fails at c =', c, ':', g(c), '≠', c * g(1))
assert c == -1, 'only the failure at c = -1 (map_neg) is emitted'
proof = tactic(
    'rintro ⟨B, hB⟩\n'
    'set e : EuclideanSpace ℝ (Fin 2) := EuclideanSpace.single 0 1\n'
    'have h1 := hB (-e) e\n'
    'rw [map_neg, LinearMap.neg_apply, hB, norm_neg] at h1\n'
    'have he : ‖e‖ = 1 := by simp [e]\n'
    'rw [he] at h1\n'
    'norm_num at h1'
)
