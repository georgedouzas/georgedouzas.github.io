import Mathlib
import Veriphy.Kit.Levels

/-!
# Real analysis, part "Real numbers, sequences and limits"

The claims of the section "Real numbers, sequences and limits" of `tutorial/part-i-03-real-analysis.qmd`: suprema
and the completeness axiom, limits of sequences in the `ε`-`N` form, Cauchy sequences, monotone convergence and the
limits of the running sequences `1/n`, `(1 + 1/n)^n`, `n^(1/n)` and `H_n - log n`.

A closed example (a value with no free parameter) is stated through the general claim it instantiates (a family
with the example's numbers as parameters, or the `ε`-`N` statement with `ε` free), so that it can be tested at the
example's values.
-/

namespace KaluzaKlein.RealAnalysis.Sequences

open Filter Topology Real Set
open scoped Nat

/-! ## def-supremum -/

/-- The completeness axiom: a non-empty set of reals bounded above has a least upper bound. -/
theorem completeness (S : Set ℝ) (hne : S.Nonempty) (hb : BddAbove S) : ∃ M, IsLUB S M :=
  Real.exists_isLUB hne hb

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.completeness (({1, 4, 2} : Set ℝ)) (⟨1, by simp⟩) (⟨4, fun x hx => by simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hx; rcases hx with rfl | rfl | rfl <;> norm_num⟩) := by
  exact ⟨4, fun x hx => by simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hx; rcases hx with rfl | rfl | rfl <;> norm_num, fun b hb => hb (by simp)⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.completeness (Set.Icc (0 : ℝ) 1) (⟨0, by norm_num⟩) (bddAbove_Icc) := by
  exact ⟨1, isLUB_Icc (by norm_num)⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.completeness (({0} : Set ℝ)) (⟨0, rfl⟩) (bddAbove_singleton) := by
  exact ⟨0, isLUB_singleton⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.completeness (Set.Ico (-1 : ℝ) 2) (⟨0, by norm_num⟩) (bddAbove_Ico) := by
  exact ⟨2, isLUB_Ico (by norm_num)⟩

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Sequences.completeness
-- test: ra-completeness-test.sage

/-- The infimum is the negated supremum of the negated set: `inf S = -sup(-S)`. -/
theorem sInf_eq_neg_sSup_neg (S : Set ℝ) : sInf S = -sSup (-S) :=
  Real.sInf_def S

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.sInf_eq_neg_sSup_neg ((∅ : Set ℝ)) := by
  simp

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.sInf_eq_neg_sSup_neg (Set.Icc (0 : ℝ) 1) := by
  rw [Set.neg_Icc, csInf_Icc (by norm_num), csSup_Icc (by norm_num)]; norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.sInf_eq_neg_sSup_neg (Set.Icc (-2 : ℝ) 3) := by
  rw [Set.neg_Icc, csInf_Icc (by norm_num), csSup_Icc (by norm_num)]; norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.sInf_eq_neg_sSup_neg (Set.Icc (1 : ℝ) 1) := by
  rw [Set.neg_Icc, csInf_Icc (by norm_num), csSup_Icc (by norm_num)]; norm_num

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Sequences.sInf_eq_neg_sSup_neg
-- test: ra-sinf-eq-neg-ssup-neg-test.sage

/-! ## exm-sup-finite -/

/-- A finite set `{a, b, c}` with largest element `b` has supremum `b`, attained (a maximum). -/
theorem isGreatest_three (a b c : ℝ) (hab : a ≤ b) (hcb : c ≤ b) :
    IsGreatest {a, b, c} b ∧ sSup {a, b, c} = b := by
  have h : IsGreatest {a, b, c} b := by
    refine ⟨by simp, fun x hx => ?_⟩
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hx
    rcases hx with rfl | rfl | rfl
    · exact hab
    · exact le_rfl
    · exact hcb
  exact ⟨h, h.csSup_eq⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.isGreatest_three (1) (4) (2) ((by norm_num)) ((by norm_num)) := by
  refine ⟨⟨by simp, fun x hx => ?_⟩, IsGreatest.csSup_eq ⟨by simp, fun x hx => ?_⟩⟩ <;> (simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hx; rcases hx with rfl | rfl | rfl <;> norm_num)

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.isGreatest_three (0) (0) (0) ((by norm_num)) ((by norm_num)) := by
  refine ⟨⟨by simp, fun x hx => ?_⟩, IsGreatest.csSup_eq ⟨by simp, fun x hx => ?_⟩⟩ <;> (simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hx; rcases hx with rfl | rfl | rfl <;> norm_num)

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.isGreatest_three (-1) (3 / 2) (2 / 3) ((by norm_num)) ((by norm_num)) := by
  refine ⟨⟨by simp, fun x hx => ?_⟩, IsGreatest.csSup_eq ⟨by simp, fun x hx => ?_⟩⟩ <;> (simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hx; rcases hx with rfl | rfl | rfl <;> norm_num)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Sequences.isGreatest_three
-- test: ra-isgreatest-three-test.sage

/-- `sup [a, b) = b`, which is not an element of the set: a supremum need not be attained. -/
theorem sSup_Ico_not_mem (a b : ℝ) (h : a < b) :
    IsLUB (Set.Ico a b) b ∧ sSup (Set.Ico a b) = b ∧ b ∉ Set.Ico a b :=
  ⟨isLUB_Ico h, csSup_Ico h, by simp⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.sSup_Ico_not_mem (0) (1) ((by norm_num)) := by
  exact ⟨isLUB_Ico (by norm_num), csSup_Ico (by norm_num), by norm_num⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.sSup_Ico_not_mem (-2) (3) ((by norm_num)) := by
  exact ⟨isLUB_Ico (by norm_num), csSup_Ico (by norm_num), by norm_num⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.sSup_Ico_not_mem (1 / 2) (3 / 4) ((by norm_num)) := by
  exact ⟨isLUB_Ico (by norm_num), csSup_Ico (by norm_num), by norm_num⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Sequences.sSup_Ico_not_mem
-- test: ra-ssup-ico-not-mem-test.sage

/-- `inf (a, ∞) = a`, and `(a, ∞)` is unbounded above, so it has no supremum. -/
theorem sInf_Ioi_unbounded (a : ℝ) :
    IsGLB (Set.Ioi a) a ∧ sInf (Set.Ioi a) = a ∧ ¬ BddAbove (Set.Ioi a) :=
  ⟨isGLB_Ioi, csInf_Ioi, not_bddAbove_Ioi a⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.sInf_Ioi_unbounded (0) := by
  exact ⟨isGLB_Ioi, csInf_Ioi, not_bddAbove_Ioi _⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.sInf_Ioi_unbounded (1) := by
  exact ⟨isGLB_Ioi, csInf_Ioi, not_bddAbove_Ioi _⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.sInf_Ioi_unbounded (-5 / 2) := by
  exact ⟨isGLB_Ioi, csInf_Ioi, not_bddAbove_Ioi _⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Sequences.sInf_Ioi_unbounded
-- test: ra-sinf-ioi-unbounded-test.sage

/-! ## exm-sup-sequence-set -/

/-- `sup {1 - 1/n} = 1` in the `ε` form: every element is below `1`, and for `n > 1/ε` the element `1 - 1/n`
exceeds `1 - ε`. -/
theorem one_sub_inv_sup (ε : ℝ) (hε : 0 < ε) :
    (∀ n : ℕ, 1 ≤ n → 1 - 1 / (n : ℝ) < 1) ∧ ∀ n : ℕ, 1 / ε < n → 1 - ε < 1 - 1 / (n : ℝ) := by
  refine ⟨fun n hn => ?_, fun n hn => ?_⟩
  · have h1 : (1 : ℝ) ≤ n := by exact_mod_cast hn
    have : 0 < 1 / (n : ℝ) := by positivity
    linarith
  · have hn0 : (0 : ℝ) < n := lt_trans (by positivity) hn
    have : 1 / (n : ℝ) < ε := by
      rw [div_lt_iff₀ hn0]
      rw [div_lt_iff₀ hε] at hn
      linarith
    linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.one_sub_inv_sup (1) ((by norm_num)) := by
  refine ⟨fun n hn => ?_, fun n hn => ?_⟩
  · have h1 : (1 : ℝ) ≤ n := by exact_mod_cast hn
    have : 0 < 1 / (n : ℝ) := by positivity
    linarith
  · have hn0 : (0 : ℝ) < n := lt_trans (by positivity) hn
    rw [sub_lt_sub_iff_left, div_lt_iff₀ hn0]
    rw [div_lt_iff₀ (by norm_num)] at hn
    linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.one_sub_inv_sup (1 / 10) ((by norm_num)) := by
  refine ⟨fun n hn => ?_, fun n hn => ?_⟩
  · have h1 : (1 : ℝ) ≤ n := by exact_mod_cast hn
    have : 0 < 1 / (n : ℝ) := by positivity
    linarith
  · have hn0 : (0 : ℝ) < n := lt_trans (by positivity) hn
    rw [sub_lt_sub_iff_left, div_lt_iff₀ hn0]
    rw [div_lt_iff₀ (by norm_num)] at hn
    linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.one_sub_inv_sup (1 / 1000) ((by norm_num)) := by
  refine ⟨fun n hn => ?_, fun n hn => ?_⟩
  · have h1 : (1 : ℝ) ≤ n := by exact_mod_cast hn
    have : 0 < 1 / (n : ℝ) := by positivity
    linarith
  · have hn0 : (0 : ℝ) < n := lt_trans (by positivity) hn
    rw [sub_lt_sub_iff_left, div_lt_iff₀ hn0]
    rw [div_lt_iff₀ (by norm_num)] at hn
    linarith

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Sequences.one_sub_inv_sup
-- test: ra-one-sub-inv-sup-test.sage

/-! ## exm-sup-rationals -/

/-- `√p` is irrational for a prime `p` (for `p = 2`, the example). -/
theorem sqrt_prime_irrational (p : ℕ) (hp : p.Prime) : Irrational (√(p : ℝ)) :=
  hp.irrational_sqrt

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.sqrt_prime_irrational (2) (Nat.prime_two) := by
  exact Nat.Prime.irrational_sqrt (by norm_num)

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.sqrt_prime_irrational (3) (Nat.prime_three) := by
  exact Nat.Prime.irrational_sqrt (by norm_num)

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.sqrt_prime_irrational (5) ((by norm_num)) := by
  exact Nat.Prime.irrational_sqrt (by norm_num)

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Sequences.sqrt_prime_irrational
-- test: ra-sqrt-prime-irrational-test.sage

/-- `{x ∈ ℝ : x² < p}` has supremum `√p` and infimum `-√p`, neither attained. -/
theorem sq_lt_bounds (p : ℝ) (hp : 0 < p) :
    IsLUB {x : ℝ | x ^ 2 < p} (√p) ∧ IsGLB {x : ℝ | x ^ 2 < p} (-√p) ∧
      √p ∉ {x : ℝ | x ^ 2 < p} ∧ -√p ∉ {x : ℝ | x ^ 2 < p} := by
  have hs := Real.sq_sqrt hp.le
  have h0 : 0 < √p := Real.sqrt_pos.2 hp
  have mem : ∀ x : ℝ, x ^ 2 < p ↔ -√p < x ∧ x < √p := by
    intro x
    constructor
    · intro hx
      constructor <;> nlinarith [sq_nonneg (x + √p), sq_nonneg (x - √p)]
    · rintro ⟨h1, h2⟩
      nlinarith
  have hset : {x : ℝ | x ^ 2 < p} = Set.Ioo (-√p) (√p) := by
    ext x; simp [mem x]
  rw [hset]
  have hlt : -√p < √p := by linarith
  exact ⟨isLUB_Ioo hlt, isGLB_Ioo hlt, by simp, by simp⟩

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Sequences.sq_lt_bounds
-- test: ra-sq-lt-bounds-test.sage

/-- The rationals are not complete: `{q ∈ ℚ : q² < p}` is bounded above in `ℚ` but has no rational least upper
bound, for a prime `p` (for `p = 2`, the example). -/
theorem no_rat_lub (p : ℕ) (hp : p.Prime) :
    BddAbove {q : ℚ | q ^ 2 < p} ∧ ¬ ∃ q : ℚ, IsLUB {r : ℚ | r ^ 2 < p} q := by
  have hirr : Irrational (√(p : ℝ)) := hp.irrational_sqrt
  have hp0 : (0 : ℝ) < p := by exact_mod_cast hp.pos
  have hs := Real.sq_sqrt hp0.le
  have hsq0 := Real.sqrt_nonneg (p : ℝ)
  refine ⟨⟨p, fun r hr => ?_⟩, ?_⟩
  · simp only [Set.mem_setOf_eq] at hr
    have h1 : (1 : ℚ) ≤ p := by exact_mod_cast hp.one_lt.le
    by_contra h
    push_neg at h
    nlinarith
  · rintro ⟨q, hq⟩
    have hp0q : (0 : ℚ) < p := by exact_mod_cast hp.pos
    have hq0 : (0 : ℚ) ≤ q := hq.1 (show (0 : ℚ) ^ 2 < p by nlinarith)
    rcases lt_trichotomy (q : ℝ) (√p) with h | h | h
    · obtain ⟨r, hr1, hr2⟩ := exists_rat_btwn h
      have hr0 : (0 : ℝ) ≤ r := le_trans (by exact_mod_cast hq0) hr1.le
      have hmem : r ∈ {r : ℚ | r ^ 2 < p} := by
        simp only [Set.mem_setOf_eq]
        have : ((r : ℝ)) ^ 2 < p := by nlinarith
        exact_mod_cast this
      have h2 : (r : ℝ) ≤ q := by exact_mod_cast hq.1 hmem
      linarith
    · exact hirr ⟨q, h⟩
    · obtain ⟨r, hr1, hr2⟩ := exists_rat_btwn h
      have hub : r ∈ upperBounds {r : ℚ | r ^ 2 < p} := by
        intro s hs'
        simp only [Set.mem_setOf_eq] at hs'
        have h3 : ((s : ℝ)) ^ 2 < p := by exact_mod_cast hs'
        have h4 : (s : ℝ) < √p := by
          by_contra hc
          push_neg at hc
          nlinarith
        exact_mod_cast (lt_trans h4 hr1).le
      have h5 : (q : ℝ) ≤ r := by exact_mod_cast hq.2 hub
      linarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Sequences.no_rat_lub
-- test: ra-no-rat-lub-test.sage

/-! ## exr-sup-examples -/

/-- The set `{(-1)^n + 1/n : n ≥ 1}`: every element lies in `(-1, 3/2]`, and `3/2` is attained at `n = 2`. -/
theorem alt_bounds (n : ℕ) (hn : 1 ≤ n) :
    -1 < (-1 : ℝ) ^ n + 1 / n ∧ (-1 : ℝ) ^ n + 1 / n ≤ 3 / 2 ∧ (n = 2 → (-1 : ℝ) ^ n + 1 / n = 3 / 2) := by
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast hn
  have hpos : 0 < 1 / (n : ℝ) := by positivity
  refine ⟨?_, ?_, fun h => by subst h; norm_num⟩
  · rcases neg_one_pow_eq_or ℝ n with h | h <;> rw [h] <;> linarith
  · rcases Nat.even_or_odd n with he | ho
    · rw [he.neg_one_pow]
      have h2 : (2 : ℝ) ≤ n := by
        rcases he with ⟨k, hk⟩
        have : 1 ≤ k := by omega
        have : (2 : ℕ) ≤ n := by omega
        exact_mod_cast this
      have : 1 / (n : ℝ) ≤ 1 / 2 := by
        rw [div_le_div_iff₀ (by positivity) (by norm_num)]; linarith
      linarith
    · rw [ho.neg_one_pow]
      have : 1 / (n : ℝ) ≤ 1 := by rw [div_le_one (by positivity)]; exact hn1
      linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.alt_bounds (1) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.alt_bounds (2) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.alt_bounds (3) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.alt_bounds (7) ((by norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Sequences.alt_bounds
-- test: ra-alt-bounds-test.sage

/-- The infimum `-1` of `{(-1)^n + 1/n}` is approached by the odd terms: for every `ε > 0` some odd `n` has
`(-1)^n + 1/n < -1 + ε`. -/
theorem alt_inf (ε : ℝ) (hε : 0 < ε) : ∃ n : ℕ, 1 ≤ n ∧ (-1 : ℝ) ^ n + 1 / n < -1 + ε := by
  obtain ⟨k, hk⟩ := exists_nat_gt (1 / ε)
  refine ⟨2 * k + 1, by omega, ?_⟩
  rw [(odd_two_mul_add_one k).neg_one_pow]
  have hk0 : (0 : ℝ) < k := lt_trans (by positivity) hk
  have h1 : 1 / ((2 * k + 1 : ℕ) : ℝ) < ε := by
    push_cast
    rw [div_lt_iff₀ (by positivity)]
    rw [div_lt_iff₀ hε] at hk
    nlinarith
  linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.alt_inf (1) ((by norm_num)) := by
  exact ⟨3, by norm_num, by norm_num⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.alt_inf (1 / 10) ((by norm_num)) := by
  exact ⟨11, by norm_num, by norm_num⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.alt_inf (1 / 1000) ((by norm_num)) := by
  exact ⟨1001, by norm_num, by norm_num⟩

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Sequences.alt_inf
-- test: ra-alt-inf-test.sage

/-- No element of `{sin n : n ∈ ℕ}` equals `1`: `sin n = 1` would make `π` rational. -/
theorem sin_nat_lt_one (n : ℕ) : Real.sin n < 1 := by
  rcases (Real.sin_le_one (n : ℝ)).lt_or_eq with h | h
  · exact h
  · exfalso
    rw [Real.sin_eq_one_iff] at h
    obtain ⟨k, hk⟩ := h
    have hden : (1 / 2 + 2 * (k : ℝ)) ≠ 0 := by
      intro h0
      have : (4 * k + 1 : ℤ) = 0 := by
        have : ((4 * k + 1 : ℤ) : ℝ) = 0 := by push_cast; linarith
        exact_mod_cast this
      omega
    have hpi : π = (n : ℝ) / (1 / 2 + 2 * k) := by
      rw [eq_div_iff hden]
      linear_combination hk
    apply irrational_pi
    refine ⟨(n : ℚ) / (1 / 2 + 2 * k), ?_⟩
    rw [hpi]
    push_cast
    ring

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.RealAnalysis.Sequences.sin_nat_lt_one
-- test: ra-sin-nat-lt-one-test.sage

/-- No element of `{sin n : n ∈ ℕ}` equals `-1`. -/
theorem neg_one_lt_sin_nat (n : ℕ) : -1 < Real.sin n := by
  rcases (Real.neg_one_le_sin (n : ℝ)).lt_or_eq with h | h
  · exact h
  · exfalso
    rw [eq_comm, Real.sin_eq_neg_one_iff] at h
    obtain ⟨k, hk⟩ := h
    have hden : (-1 / 2 + 2 * (k : ℝ)) ≠ 0 := by
      intro h0
      have : (4 * k - 1 : ℤ) = 0 := by
        have : ((4 * k - 1 : ℤ) : ℝ) = 0 := by push_cast; linarith
        exact_mod_cast this
      omega
    have hpi : π = (n : ℝ) / (-1 / 2 + 2 * k) := by
      rw [eq_div_iff hden]
      linear_combination hk
    apply irrational_pi
    refine ⟨(n : ℚ) / (-1 / 2 + 2 * k), ?_⟩
    rw [hpi]
    push_cast
    ring

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Sequences.neg_one_lt_sin_nat
-- test: ra-neg-one-lt-sin-nat-test.sage

/-! ## exr-sup-characterisation -/

/-- The `ε`-characterisation of the supremum: `M = sup S` if and only if `M` is an upper bound and every
`M - ε` is exceeded by an element of `S`. -/
theorem isLUB_iff_eps (S : Set ℝ) (M : ℝ) :
    IsLUB S M ↔ M ∈ upperBounds S ∧ ∀ ε > 0, ∃ s ∈ S, M - ε < s := by
  constructor
  · intro h
    refine ⟨h.1, fun ε hε => ?_⟩
    exact (lt_isLUB_iff h).1 (by linarith)
  · rintro ⟨hub, heps⟩
    refine ⟨hub, fun M' hM' => ?_⟩
    by_contra hlt
    push_neg at hlt
    obtain ⟨s, hs, hs'⟩ := heps (M - M') (by linarith)
    have := hM' hs
    linarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Sequences.isLUB_iff_eps
-- test: ra-islub-iff-eps-test.sage

/-! ## exr-supremum-sage -/

/-- `{1/n : n ≥ 1}` has the maximum `1` (attained at `n = 1`) and the infimum `0`, not attained: every element is
in `(0, 1]` and for `n > 1/ε` the element `1/n` is below `ε`. -/
theorem inv_nat_sup_inf (ε : ℝ) (hε : 0 < ε) :
    (∀ n : ℕ, 1 ≤ n → 0 < 1 / (n : ℝ) ∧ 1 / (n : ℝ) ≤ 1) ∧ 1 / ((1 : ℕ) : ℝ) = 1 ∧
      ∀ n : ℕ, 1 / ε < n → 1 / (n : ℝ) < ε := by
  refine ⟨fun n hn => ?_, by norm_num, fun n hn => ?_⟩
  · have h1 : (1 : ℝ) ≤ n := by exact_mod_cast hn
    exact ⟨by positivity, by rw [div_le_one (by positivity)]; exact h1⟩
  · have hn0 : (0 : ℝ) < n := lt_trans (by positivity) hn
    rw [div_lt_iff₀ hn0]
    rw [div_lt_iff₀ hε] at hn
    linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.inv_nat_sup_inf (1) ((by norm_num)) := by
  refine ⟨fun n hn => ?_, by norm_num, fun n hn => ?_⟩
  · have h1 : (1 : ℝ) ≤ n := by exact_mod_cast hn
    exact ⟨by positivity, by rw [div_le_one (by positivity)]; exact h1⟩
  · have hn0 : (0 : ℝ) < n := lt_trans (by positivity) hn
    rw [div_lt_iff₀ hn0]
    rw [div_lt_iff₀ (by norm_num)] at hn
    linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.inv_nat_sup_inf (1 / 10) ((by norm_num)) := by
  refine ⟨fun n hn => ?_, by norm_num, fun n hn => ?_⟩
  · have h1 : (1 : ℝ) ≤ n := by exact_mod_cast hn
    exact ⟨by positivity, by rw [div_le_one (by positivity)]; exact h1⟩
  · have hn0 : (0 : ℝ) < n := lt_trans (by positivity) hn
    rw [div_lt_iff₀ hn0]
    rw [div_lt_iff₀ (by norm_num)] at hn
    linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.inv_nat_sup_inf (1 / 1000) ((by norm_num)) := by
  refine ⟨fun n hn => ?_, by norm_num, fun n hn => ?_⟩
  · have h1 : (1 : ℝ) ≤ n := by exact_mod_cast hn
    exact ⟨by positivity, by rw [div_le_one (by positivity)]; exact h1⟩
  · have hn0 : (0 : ℝ) < n := lt_trans (by positivity) hn
    rw [div_lt_iff₀ hn0]
    rw [div_lt_iff₀ (by norm_num)] at hn
    linarith

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Sequences.inv_nat_sup_inf
-- test: ra-inv-nat-sup-inf-test.sage

/-! ## def-sequence-limit -/

/-- Limits are unique. -/
theorem limit_unique (a : ℕ → ℝ) (L L' : ℝ) (h : Tendsto a atTop (𝓝 L)) (h' : Tendsto a atTop (𝓝 L')) :
    L = L' :=
  tendsto_nhds_unique h h'

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Sequences.limit_unique
-- test: ra-limit-unique-test.sage

/-- The sum rule: `a_n → L` and `b_n → M` imply `a_n + b_n → L + M`. -/
theorem limit_add (a b : ℕ → ℝ) (L M : ℝ) (ha : Tendsto a atTop (𝓝 L)) (hb : Tendsto b atTop (𝓝 M)) :
    Tendsto (fun n => a n + b n) atTop (𝓝 (L + M)) :=
  ha.add hb

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Sequences.limit_add
-- test: ra-limit-add-test.sage

/-- The product rule: `a_n b_n → L M`. -/
theorem limit_mul (a b : ℕ → ℝ) (L M : ℝ) (ha : Tendsto a atTop (𝓝 L)) (hb : Tendsto b atTop (𝓝 M)) :
    Tendsto (fun n => a n * b n) atTop (𝓝 (L * M)) :=
  ha.mul hb

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Sequences.limit_mul
-- test: ra-limit-mul-test.sage

/-- The quotient rule: `a_n / b_n → L / M` when `M ≠ 0`. -/
theorem limit_div (a b : ℕ → ℝ) (L M : ℝ) (ha : Tendsto a atTop (𝓝 L)) (hb : Tendsto b atTop (𝓝 M))
    (hM : M ≠ 0) : Tendsto (fun n => a n / b n) atTop (𝓝 (L / M)) :=
  ha.div hb hM

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Sequences.limit_div
-- test: ra-limit-div-test.sage

/-- The `ε`-`N` definition of the limit is mathlib's `Tendsto … (𝓝 L)`. -/
theorem tendsto_iff_eps (a : ℕ → ℝ) (L : ℝ) :
    Tendsto a atTop (𝓝 L) ↔ ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - L| < ε := by
  rw [Metric.tendsto_atTop]
  simp only [Real.dist_eq]

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Sequences.tendsto_iff_eps
-- test: ra-tendsto-iff-eps-test.sage

/-! ## exm-limit-reciprocal -/

/-- `1/n → 0`: for `n > 1/ε`, `|1/n - 0| < ε`, and the same `N` works for `(-1)^n/n`. -/
theorem recip_eps (ε : ℝ) (hε : 0 < ε) (n : ℕ) (hn : 1 / ε < n) :
    |1 / (n : ℝ) - 0| < ε ∧ |(-1 : ℝ) ^ n / n - 0| < ε := by
  have hn0 : (0 : ℝ) < n := lt_trans (by positivity) hn
  have h1 : 1 / (n : ℝ) < ε := by
    rw [div_lt_iff₀ hn0]
    rw [div_lt_iff₀ hε] at hn
    linarith
  refine ⟨by rw [sub_zero, abs_of_pos (by positivity)]; exact h1, ?_⟩
  rw [sub_zero, abs_div, abs_pow, abs_neg, abs_one, one_pow, abs_of_pos hn0]
  exact h1

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.recip_eps (1) ((by norm_num)) (2) ((by norm_num)) := by
  constructor <;> norm_num [abs_div]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.recip_eps (1 / 10) ((by norm_num)) (11) ((by norm_num)) := by
  constructor <;> norm_num [abs_div]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.recip_eps (1 / 1000) ((by norm_num)) (1001) ((by norm_num)) := by
  constructor <;> norm_num [abs_div]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.recip_eps (1 / 10) ((by norm_num)) (12) ((by norm_num)) := by
  constructor <;> norm_num [abs_div]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Sequences.recip_eps
-- test: ra-recip-eps-test.sage

/-! ## exm-limit-rational-function -/

/-- `(2n + 1)/(n + 3) → 2` from the definition: `|a_n - 2| = 5/(n + 3) < ε` as soon as `n > 5/ε - 3`. -/
theorem rat_fun_eps (ε : ℝ) (hε : 0 < ε) (n : ℕ) (hn : 5 / ε - 3 < n) :
    |(2 * (n : ℝ) + 1) / (n + 3) - 2| = 5 / (n + 3) ∧ |(2 * (n : ℝ) + 1) / (n + 3) - 2| < ε := by
  have hn3 : (0 : ℝ) < n + 3 := by positivity
  have e : (2 * (n : ℝ) + 1) / (n + 3) - 2 = -(5 / (n + 3)) := by
    field_simp
    ring
  have habs : |(2 * (n : ℝ) + 1) / (n + 3) - 2| = 5 / (n + 3) := by
    rw [e, abs_neg, abs_of_pos (by positivity)]
  refine ⟨habs, ?_⟩
  rw [habs, div_lt_iff₀ hn3]
  have : 5 / ε < n + 3 := by linarith
  rw [div_lt_iff₀ hε] at this
  linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.rat_fun_eps (1) ((by norm_num)) (5) ((by norm_num)) := by
  constructor <;> norm_num [abs_div]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.rat_fun_eps (1 / 10) ((by norm_num)) (48) ((by norm_num)) := by
  constructor <;> norm_num [abs_div]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.rat_fun_eps (1 / 100) ((by norm_num)) (498) ((by norm_num)) := by
  constructor <;> norm_num [abs_div]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Sequences.rat_fun_eps
-- test: ra-rat-fun-eps-test.sage

/-- The leading terms decide: for polynomials of equal degree, `P(n)/Q(n)` tends to the ratio of the leading
coefficients (`(2n + 1)/(n + 3) → 2`, `(3n² + 1)/(n² - 5) → 3`). -/
theorem poly_ratio_limit (P Q : Polynomial ℝ) (h : P.degree = Q.degree) :
    Tendsto (fun n : ℕ => P.eval (n : ℝ) / Q.eval (n : ℝ)) atTop (𝓝 (P.leadingCoeff / Q.leadingCoeff)) :=
  (Polynomial.div_tendsto_atTop_leadingCoeff_div_of_degree_eq P Q h).comp tendsto_natCast_atTop_atTop

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Sequences.poly_ratio_limit
-- test: ra-poly-ratio-limit-test.sage

/-- A numerator of lower degree gives the limit `0` (`n/(n² + 1) → 0`). -/
theorem poly_ratio_limit_zero (P Q : Polynomial ℝ) (h : P.degree < Q.degree) :
    Tendsto (fun n : ℕ => P.eval (n : ℝ) / Q.eval (n : ℝ)) atTop (𝓝 0) :=
  (Polynomial.div_tendsto_atTop_zero_of_degree_lt P Q h).comp tendsto_natCast_atTop_atTop

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Sequences.poly_ratio_limit_zero
-- test: ra-poly-ratio-limit-zero-test.sage

/-! ## exm-limit-euler -/

/-- `(1 + x/n)^n → e^x`; at `x = 1` the limit is `e`. -/
theorem one_add_div_pow_limit (x : ℝ) : Tendsto (fun n : ℕ => (1 + x / n) ^ n) atTop (𝓝 (Real.exp x)) :=
  Real.tendsto_one_add_div_pow_exp x

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.RealAnalysis.Sequences.one_add_div_pow_limit
-- test: ra-one-add-div-pow-limit-test.sage

/-- The terms of `(1 + 1/n)^n` are `(n + 1)^n / n^n`: `2, 9/4 = 2.25, 64/27 = 2.370…, 625/256 = 2.441…`. -/
theorem euler_term (n : ℕ) (hn : 1 ≤ n) : (1 + 1 / (n : ℝ)) ^ n = ((n : ℝ) + 1) ^ n / (n : ℝ) ^ n := by
  have hn0 : (n : ℝ) ≠ 0 := by positivity
  rw [← div_pow]
  congr 1
  field_simp

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.euler_term (1) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.euler_term (2) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.euler_term (3) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.euler_term (4) ((by norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Sequences.euler_term
-- test: ra-euler-term-test.sage

/-! ## exr-limit-epsilon-N -/

/-- For `a_n = (2n + 1)/(n + 3)` and a threshold `N` with `N - 1 ≤ 5/ε - 3 < N`, every `n ≥ N` has
`|a_n - 2| < ε` and `n = N - 1` does not: `N` is the smallest (`N = 48, 498, 4998` for `ε = 10⁻¹, 10⁻², 10⁻³`). -/
theorem smallest_N (ε : ℝ) (hε : 0 < ε) (N : ℕ) (hN1 : 1 ≤ N) (hN : 5 / ε - 3 < N)
    (hN' : (N : ℝ) - 1 ≤ 5 / ε - 3) :
    (∀ n : ℕ, N ≤ n → |(2 * (n : ℝ) + 1) / (n + 3) - 2| < ε) ∧
      ¬ |(2 * ((N - 1 : ℕ) : ℝ) + 1) / ((N - 1 : ℕ) + 3) - 2| < ε := by
  refine ⟨fun n hn => (rat_fun_eps ε hε n (lt_of_lt_of_le hN (by exact_mod_cast hn))).2, ?_⟩
  have hc : ((N - 1 : ℕ) : ℝ) = (N : ℝ) - 1 := by
    rw [Nat.cast_sub hN1]; simp
  have hpos : (0 : ℝ) < ((N - 1 : ℕ) : ℝ) + 3 := by positivity
  have e : (2 * ((N - 1 : ℕ) : ℝ) + 1) / ((N - 1 : ℕ) + 3) - 2 = -(5 / (((N - 1 : ℕ) : ℝ) + 3)) := by
    field_simp
    ring
  rw [e, abs_neg, abs_of_pos (by positivity), not_lt, le_div_iff₀ hpos, hc]
  have := hN'
  rw [le_sub_iff_add_le, le_div_iff₀ hε] at this
  nlinarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.smallest_N (1 / 10) ((by norm_num)) (48) ((by norm_num)) ((by norm_num)) ((by norm_num)) := by
  refine ⟨fun n hn => ?_, ?_⟩
  · have h : (48 : ℝ) ≤ n := by exact_mod_cast hn
    have e : (2 * (n : ℝ) + 1) / (n + 3) - 2 = -(5 / (n + 3)) := by field_simp; ring
    rw [e, abs_neg, abs_of_pos (by positivity), div_lt_iff₀ (by positivity)]
    linarith
  · norm_num [abs_div]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.smallest_N (1 / 100) ((by norm_num)) (498) ((by norm_num)) ((by norm_num)) ((by norm_num)) := by
  refine ⟨fun n hn => ?_, ?_⟩
  · have h : (498 : ℝ) ≤ n := by exact_mod_cast hn
    have e : (2 * (n : ℝ) + 1) / (n + 3) - 2 = -(5 / (n + 3)) := by field_simp; ring
    rw [e, abs_neg, abs_of_pos (by positivity), div_lt_iff₀ (by positivity)]
    linarith
  · norm_num [abs_div]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.smallest_N (1 / 1000) ((by norm_num)) (4998) ((by norm_num)) ((by norm_num)) ((by norm_num)) := by
  refine ⟨fun n hn => ?_, ?_⟩
  · have h : (4998 : ℝ) ≤ n := by exact_mod_cast hn
    have e : (2 * (n : ℝ) + 1) / (n + 3) - 2 = -(5 / (n + 3)) := by field_simp; ring
    rw [e, abs_neg, abs_of_pos (by positivity), div_lt_iff₀ (by positivity)]
    linarith
  · norm_num [abs_div]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.smallest_N (3 / 2) ((by norm_num)) (1) ((by norm_num)) ((by norm_num)) ((by norm_num)) := by
  refine ⟨fun n hn => ?_, ?_⟩
  · have h : (1 : ℝ) ≤ n := by exact_mod_cast hn
    have e : (2 * (n : ℝ) + 1) / (n + 3) - 2 = -(5 / (n + 3)) := by field_simp; ring
    rw [e, abs_neg, abs_of_pos (by positivity), div_lt_iff₀ (by positivity)]
    linarith
  · norm_num [abs_div]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Sequences.smallest_N
-- test: ra-smallest-n-test.sage

/-! ## exr-limit-rules -/

/-- `n sin(x/n) → x`; at `x = 1`, `n sin(1/n) → 1`. -/
theorem n_sin_div_limit (x : ℝ) : Tendsto (fun n : ℕ => (n : ℝ) * Real.sin (x / n)) atTop (𝓝 x) := by
  have hc : Tendsto (fun n : ℕ => x * Real.sinc (x / n)) atTop (𝓝 (x * Real.sinc 0)) := by
    apply Tendsto.const_mul
    apply (Real.continuous_sinc.tendsto 0).comp
    simpa using (tendsto_const_div_atTop_nhds_zero_nat x)
  rw [Real.sinc_zero, mul_one] at hc
  apply hc.congr'
  filter_upwards [eventually_ne_atTop 0] with n hn
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  by_cases hx : x = 0
  · simp [hx]
  · rw [Real.sinc_of_ne_zero (div_ne_zero hx hn')]
    field_simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Sequences.n_sin_div_limit
-- test: ra-n-sin-div-limit-test.sage

/-- `√(n² + a n) - n → a/2` for `a ≥ 0`; at `a = 1` the limit is `1/2` (multiply by the conjugate). -/
theorem sqrt_sub_limit (a : ℝ) (ha : 0 ≤ a) :
    Tendsto (fun n : ℕ => √((n : ℝ) ^ 2 + a * n) - n) atTop (𝓝 (a / 2)) := by
  have key : ∀ n : ℕ, 0 < n → √((n : ℝ) ^ 2 + a * n) - n = a / (√(1 + a / n) + 1) := by
    intro n hn
    have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
    have h1 : √((n : ℝ) ^ 2 + a * n) = n * √(1 + a / n) := by
      rw [show (n : ℝ) ^ 2 + a * n = n ^ 2 * (1 + a / n) by field_simp]
      rw [Real.sqrt_mul (by positivity), Real.sqrt_sq hn0.le]
    rw [h1]
    have hs : 0 ≤ √(1 + a / n) := Real.sqrt_nonneg _
    have hsq : √(1 + a / n) ^ 2 = 1 + a / n := Real.sq_sqrt (by positivity)
    have hden : √(1 + a / n) + 1 ≠ 0 := by positivity
    rw [eq_div_iff hden, show ((n : ℝ) * √(1 + a / n) - n) * (√(1 + a / n) + 1) =
      (n : ℝ) * (√(1 + a / n) ^ 2 - 1) by ring, hsq]
    field_simp <;> ring
  have hlim : Tendsto (fun n : ℕ => a / (√(1 + a / n) + 1)) atTop (𝓝 (a / (√(1 + 0) + 1))) := by
    apply Tendsto.div tendsto_const_nhds _ (by positivity)
    apply Tendsto.add_const
    apply (Real.continuous_sqrt.tendsto _).comp
    exact tendsto_const_nhds.add (tendsto_const_div_atTop_nhds_zero_nat a)
  rw [add_zero, Real.sqrt_one, show (1 : ℝ) + 1 = 2 by norm_num] at hlim
  apply hlim.congr'
  filter_upwards [eventually_gt_atTop 0] with n hn
  exact (key n hn).symm

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Sequences.sqrt_sub_limit
-- test: ra-sqrt-sub-limit-test.sage

/-- The lower Stirling bound `(n/e)^n ≤ n!`, which gives `(n!)^(1/n)/n ≥ 1/e`. -/
theorem factorial_lower (n : ℕ) : ((n : ℝ) / Real.exp 1) ^ n ≤ n ! := by
  have h := Real.pow_div_factorial_le_exp (n : ℝ) (Nat.cast_nonneg n) n
  have he : Real.exp (n : ℝ) = Real.exp 1 ^ n := by
    rw [← Real.exp_nat_mul, mul_one]
  rw [he] at h
  have hf : (0 : ℝ) < n ! := by exact_mod_cast Nat.factorial_pos n
  have hpos : 0 < Real.exp 1 ^ n := by positivity
  rw [div_pow, div_le_iff₀ hpos]
  rw [div_le_iff₀ hf] at h
  linarith

/-- Stirling's bounds give `(n!)^(1/n)/n → 1/e`: `1/e ≤ (n!)^(1/n)/n ≤ (e √n)^(1/n)/e`, and `(e √n)^(1/n) → 1`. -/
theorem factorial_root_bounds (n : ℕ) (hn : 1 ≤ n) :
    Real.exp (-1) ≤ (n ! : ℝ) ^ ((n : ℝ)⁻¹) / n ∧
      (n ! : ℝ) ^ ((n : ℝ)⁻¹) / n ≤ (Real.exp 1 * √n) ^ ((n : ℝ)⁻¹) * Real.exp (-1) := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  have hne : (n : ℝ) ≠ 0 := hn0.ne'
  have hnn : n ≠ 0 := by omega
  have he : 0 < Real.exp 1 := Real.exp_pos 1
  have hf : (0 : ℝ) < n ! := by exact_mod_cast Nat.factorial_pos n
  constructor
  · -- (n/e)^n ≤ n!, so n/e ≤ (n!)^(1/n)
    have h := factorial_lower n
    have h2 : ((n : ℝ) / Real.exp 1) ≤ (n ! : ℝ) ^ ((n : ℝ)⁻¹) := by
      have := Real.rpow_le_rpow (by positivity) h (by positivity : (0 : ℝ) ≤ (n : ℝ)⁻¹)
      rwa [Real.pow_rpow_inv_natCast (by positivity) hnn] at this
    rw [le_div_iff₀ hn0, Real.exp_neg]
    calc (Real.exp 1)⁻¹ * n = n / Real.exp 1 := by field_simp
      _ ≤ _ := h2
  · -- n! = S_n √(2n) (n/e)^n ≤ (e/√2) √(2n) (n/e)^n = e √n (n/e)^n
    have hS : Stirling.stirlingSeq n ≤ Real.exp 1 / √2 := by
      rw [← Stirling.stirlingSeq_one]
      obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
      exact Stirling.stirlingSeq'_antitone (Nat.zero_le k)
    have hfact : (n ! : ℝ) = Stirling.stirlingSeq n * (√(2 * n) * (n / Real.exp 1) ^ n) := by
      unfold Stirling.stirlingSeq
      have : √(2 * (n : ℝ)) * ((n : ℝ) / Real.exp 1) ^ n ≠ 0 := by positivity
      field_simp
    have h2 : √(2 * (n : ℝ)) = √2 * √n := Real.sqrt_mul (by norm_num) _
    have hle : (n ! : ℝ) ≤ (Real.exp 1 * √n) * (n / Real.exp 1) ^ n := by
      rw [hfact, h2]
      have hs2 : 0 < √(2 : ℝ) := by positivity
      calc Stirling.stirlingSeq n * (√2 * √n * (n / Real.exp 1) ^ n)
          ≤ Real.exp 1 / √2 * (√2 * √n * (n / Real.exp 1) ^ n) := by gcongr
        _ = (Real.exp 1 * √n) * (n / Real.exp 1) ^ n := by field_simp
    have h3 := Real.rpow_le_rpow hf.le hle (by positivity : (0 : ℝ) ≤ (n : ℝ)⁻¹)
    rw [Real.mul_rpow (by positivity) (by positivity), Real.pow_rpow_inv_natCast (by positivity) hnn] at h3
    rw [div_le_iff₀ hn0, Real.exp_neg]
    calc (n ! : ℝ) ^ ((n : ℝ)⁻¹) ≤ (Real.exp 1 * √n) ^ ((n : ℝ)⁻¹) * (n / Real.exp 1) := h3
      _ = (Real.exp 1 * √n) ^ ((n : ℝ)⁻¹) * (Real.exp 1)⁻¹ * n := by field_simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Sequences.factorial_root_bounds
-- test: ra-factorial-root-bounds-test.sage

/-! ## def-cauchy-sequence -/

/-- A convergent sequence is Cauchy (in the `ε` form of the definition). -/
theorem convergent_cauchy (a : ℕ → ℝ) (L : ℝ) (h : Tendsto a atTop (𝓝 L)) (ε : ℝ) (hε : 0 < ε) :
    ∃ N : ℕ, ∀ m n : ℕ, N ≤ m → N ≤ n → |a m - a n| < ε := by
  obtain ⟨N, hN⟩ := (Metric.cauchySeq_iff.1 h.cauchySeq) ε hε
  exact ⟨N, fun m n hm hn => by simpa [Real.dist_eq] using hN m hm n hn⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.convergent_cauchy ((fun _ : ℕ => (0 : ℝ))) (0) (tendsto_const_nhds) (1) ((by norm_num)) := by
  exact ⟨0, fun m n _ _ => by norm_num⟩

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.convergent_cauchy ((fun n : ℕ => 1 / ((n : ℝ) + 1))) (0) (tendsto_one_div_add_atTop_nhds_zero_nat) (1 / 10) ((by norm_num)) := by
  refine ⟨10, fun m n hm hn => ?_⟩
  have hm' : (10 : ℝ) ≤ m := by exact_mod_cast hm
  have hn' : (10 : ℝ) ≤ n := by exact_mod_cast hn
  have h1 : 1 / ((m : ℝ) + 1) ≤ 1 / 11 := one_div_le_one_div_of_le (by norm_num) (by linarith)
  have h2 : 1 / ((n : ℝ) + 1) ≤ 1 / 11 := one_div_le_one_div_of_le (by norm_num) (by linarith)
  have h3 : 0 < 1 / ((m : ℝ) + 1) := by positivity
  have h4 : 0 < 1 / ((n : ℝ) + 1) := by positivity
  rw [abs_lt]; constructor <;> linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.convergent_cauchy ((fun n : ℕ => 1 / ((n : ℝ) + 1))) (0) (tendsto_one_div_add_atTop_nhds_zero_nat) (1 / 1000) ((by norm_num)) := by
  refine ⟨1000, fun m n hm hn => ?_⟩
  have hm' : (1000 : ℝ) ≤ m := by exact_mod_cast hm
  have hn' : (1000 : ℝ) ≤ n := by exact_mod_cast hn
  have h1 : 1 / ((m : ℝ) + 1) ≤ 1 / 1001 := one_div_le_one_div_of_le (by norm_num) (by linarith)
  have h2 : 1 / ((n : ℝ) + 1) ≤ 1 / 1001 := one_div_le_one_div_of_le (by norm_num) (by linarith)
  have h3 : 0 < 1 / ((m : ℝ) + 1) := by positivity
  have h4 : 0 < 1 / ((n : ℝ) + 1) := by positivity
  rw [abs_lt]; constructor <;> linarith

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Sequences.convergent_cauchy
-- test: ra-convergent-cauchy-test.sage

/-- The Cauchy criterion in `ℝ`: a Cauchy sequence converges. -/
theorem cauchy_converges (a : ℕ → ℝ)
    (h : ∀ ε > 0, ∃ N : ℕ, ∀ m n : ℕ, N ≤ m → N ≤ n → |a m - a n| < ε) :
    ∃ L, Tendsto a atTop (𝓝 L) := by
  apply cauchySeq_tendsto_of_complete
  rw [Metric.cauchySeq_iff]
  intro ε hε
  obtain ⟨N, hN⟩ := h ε hε
  exact ⟨N, fun m hm n hn => by simpa [Real.dist_eq] using hN m n hm hn⟩

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Sequences.cauchy_converges
-- test: ra-cauchy-converges-test.sage

/-! ## exm-cauchy-convergent -/

/-- The triangle-inequality step: terms within `ε/2` of a common limit are within `ε` of each other. -/
theorem close_to_limit_close (am an L ε : ℝ) (hm : |am - L| < ε / 2) (hn : |an - L| < ε / 2) :
    |am - an| < ε := by
  have := abs_sub_le am L an
  rw [abs_sub_comm L an] at this
  linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.close_to_limit_close (1) (1) (1) (1) ((by norm_num)) ((by norm_num)) := by
  norm_num [abs_lt]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.close_to_limit_close (21 / 20) (19 / 20) (1) (1 / 5) ((by norm_num)) ((by norm_num)) := by
  norm_num [abs_lt]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.close_to_limit_close (3) (-1 / 2) (1) (8) ((by norm_num)) ((by norm_num)) := by
  norm_num [abs_lt]

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.RealAnalysis.Sequences.close_to_limit_close
-- test: ra-close-to-limit-close-test.sage

/-! ## exm-cauchy-factorial -/

/-- The partial sums `s_n = ∑_{k ≤ n} 1/k!` of the series of `e`. -/
noncomputable def expPartial (n : ℕ) : ℝ := ∑ k ∈ Finset.range (n + 1), 1 / (k ! : ℝ)

/-- `|s_m - s_n| ≤ 2/(n + 1)!` for `m > n`: the tail is bounded by a geometric series. -/
theorem expPartial_tail (n m : ℕ) (h : n < m) : |expPartial m - expPartial n| ≤ 2 / ((n + 1) ! : ℝ) := by
  have hmono : expPartial n ≤ expPartial m := by
    unfold expPartial
    apply Finset.sum_le_sum_of_subset_of_nonneg
    · exact Finset.range_subset_range.2 (by omega)
    · intro i _ _; positivity
  have hle : expPartial m ≤ Real.exp 1 := by
    have := Real.sum_le_exp_of_nonneg (zero_le_one' ℝ) (m + 1)
    simpa [expPartial] using this
  have hb := Real.exp_bound (x := 1) (by norm_num) (n := n + 1) (by omega)
  simp only [abs_one, one_pow, one_mul] at hb
  have hb' : Real.exp 1 - expPartial n ≤ ((n + 1).succ : ℝ) / ((n + 1) ! * (n + 1 : ℕ)) := by
    have := (abs_le.1 hb).2
    simpa [expPartial] using this
  rw [abs_of_nonneg (by linarith)]
  have hf : (0 : ℝ) < (n + 1) ! := by exact_mod_cast Nat.factorial_pos _
  have key : ((n + 1).succ : ℝ) / ((n + 1) ! * (n + 1 : ℕ)) ≤ 2 / ((n + 1) ! : ℝ) := by
    rw [div_le_div_iff₀ (by positivity) hf]
    push_cast
    nlinarith
  linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.expPartial_tail (0) (1) ((by norm_num)) := by
  simp only [KaluzaKlein.RealAnalysis.Sequences.expPartial, Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial]
  norm_num [abs_le]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.expPartial_tail (2) (5) ((by norm_num)) := by
  simp only [KaluzaKlein.RealAnalysis.Sequences.expPartial, Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial]
  norm_num [abs_le]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.expPartial_tail (4) (9) ((by norm_num)) := by
  simp only [KaluzaKlein.RealAnalysis.Sequences.expPartial, Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial]
  norm_num [abs_le]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.expPartial_tail (6) (20) ((by norm_num)) := by
  simp only [KaluzaKlein.RealAnalysis.Sequences.expPartial, Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial]
  norm_num [abs_le]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.expPartial_tail (8) (27) ((by norm_num)) := by
  simp only [KaluzaKlein.RealAnalysis.Sequences.expPartial, Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial]
  norm_num [abs_le]

attribute [veriphy_tested kernel_test 5 0] KaluzaKlein.RealAnalysis.Sequences.expPartial_tail
-- test: ra-exppartial-tail-test.sage

/-- The exponential series: `∑ x^k/k! = e^x`; at `x = 1` the sum is `e`. -/
theorem exp_series (x : ℝ) : HasSum (fun k : ℕ => x ^ k / (k ! : ℝ)) (Real.exp x) := by
  rw [Real.exp_eq_exp_ℝ]
  exact NormedSpace.expSeries_div_hasSum_exp x

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.RealAnalysis.Sequences.exp_series
-- test: ra-exp-series-test.sage

/-! ## exm-cauchy-not-convergent-rationals -/

/-- The decimal truncation `⌊10ⁿ x⌋ / 10ⁿ` of a real number, a rational. -/
noncomputable def decimalTrunc (x : ℝ) (n : ℕ) : ℚ := (⌊x * 10 ^ n⌋ : ℚ) / 10 ^ n

theorem decimalTrunc_le (x : ℝ) (n : ℕ) : (decimalTrunc x n : ℝ) ≤ x ∧ x < decimalTrunc x n + 1 / 10 ^ n := by
  have hp : (0 : ℝ) < 10 ^ n := by positivity
  unfold decimalTrunc
  push_cast
  constructor
  · rw [div_le_iff₀ hp]; exact Int.floor_le _
  · rw [← add_div, lt_div_iff₀ hp]; exact Int.lt_floor_add_one _

theorem decimalTrunc_mono (x : ℝ) {n m : ℕ} (h : n ≤ m) : (decimalTrunc x n : ℝ) ≤ decimalTrunc x m := by
  have hp : (0 : ℝ) < 10 ^ n := by positivity
  have hq : (0 : ℝ) < 10 ^ m := by positivity
  unfold decimalTrunc
  push_cast
  rw [div_le_div_iff₀ hp hq]
  have hk : (10 : ℝ) ^ m = 10 ^ n * 10 ^ (m - n) := by rw [← pow_add]; congr 1; omega
  -- ⌊x 10ⁿ⌋ 10^(m-n) ≤ ⌊x 10^m⌋
  have h1 : ((⌊x * 10 ^ n⌋ * 10 ^ (m - n) : ℤ) : ℝ) ≤ x * 10 ^ m := by
    push_cast
    rw [hk, ← mul_assoc]
    exact mul_le_mul_of_nonneg_right (Int.floor_le _) (by positivity)
  have h2 : (⌊x * 10 ^ n⌋ * 10 ^ (m - n) : ℤ) ≤ ⌊x * 10 ^ m⌋ := Int.le_floor.2 h1
  have h3 : ((⌊x * 10 ^ n⌋ * 10 ^ (m - n) : ℤ) : ℝ) ≤ (⌊x * 10 ^ m⌋ : ℝ) := by exact_mod_cast h2
  push_cast at h3
  calc (⌊x * 10 ^ n⌋ : ℝ) * 10 ^ m = ((⌊x * 10 ^ n⌋ : ℝ) * 10 ^ (m - n)) * 10 ^ n := by rw [hk]; ring
    _ ≤ (⌊x * 10 ^ m⌋ : ℝ) * 10 ^ n := mul_le_mul_of_nonneg_right h3 hp.le

/-- The decimal approximations form a Cauchy sequence: `|a_m - a_n| < 10⁻ⁿ` for `m ≥ n`. -/
theorem decimalTrunc_cauchy (x : ℝ) (n m : ℕ) (h : n ≤ m) :
    |(decimalTrunc x m : ℝ) - decimalTrunc x n| < 1 / 10 ^ n := by
  have h1 := decimalTrunc_le x n
  have h2 := decimalTrunc_le x m
  have h3 := decimalTrunc_mono x h
  rw [abs_of_nonneg (by linarith)]
  linarith

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.RealAnalysis.Sequences.decimalTrunc_cauchy
-- test: ra-decimaltrunc-cauchy-test.sage

/-- The decimal approximations of an irrational number (of `√2`) converge to it in `ℝ` and have no rational
limit: the Cauchy criterion fails in `ℚ`. -/
theorem decimalTrunc_no_rat_limit (x : ℝ) (hx : Irrational x) :
    Tendsto (fun n => (decimalTrunc x n : ℝ)) atTop (𝓝 x) ∧
      ¬ ∃ q : ℚ, Tendsto (fun n => (decimalTrunc x n : ℝ)) atTop (𝓝 (q : ℝ)) := by
  have hlim : Tendsto (fun n => (decimalTrunc x n : ℝ)) atTop (𝓝 x) := by
    have hu : Tendsto (fun n : ℕ => x - 1 / (10 : ℝ) ^ n) atTop (𝓝 (x - 0)) := by
      apply tendsto_const_nhds.sub
      simp_rw [one_div, ← inv_pow]
      exact tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
    rw [sub_zero] at hu
    apply tendsto_of_tendsto_of_tendsto_of_le_of_le hu tendsto_const_nhds
    · intro n; have := (decimalTrunc_le x n).2; simp only; linarith
    · intro n; exact (decimalTrunc_le x n).1
  refine ⟨hlim, ?_⟩
  rintro ⟨q, hq⟩
  exact hx ⟨q, tendsto_nhds_unique hq hlim⟩

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Sequences.decimalTrunc_no_rat_limit
-- test: ra-decimaltrunc-no-rat-limit-test.sage

/-! ## exr-cauchy-geometric -/

/-- The partial sums `a_n = ∑_{k ≤ n} r^k` satisfy `|a_m - a_n| ≤ |r|^(n+1)/(1 - |r|)` for `m > n`, `|r| < 1`. -/
theorem geom_tail (r : ℝ) (hr : |r| < 1) (n m : ℕ) (h : n < m) :
    |∑ k ∈ Finset.range (m + 1), r ^ k - ∑ k ∈ Finset.range (n + 1), r ^ k| ≤ |r| ^ (n + 1) / (1 - |r|) := by
  have hsplit : ∑ k ∈ Finset.range (m + 1), r ^ k - ∑ k ∈ Finset.range (n + 1), r ^ k =
      ∑ k ∈ Finset.Ico (n + 1) (m + 1), r ^ k := by
    rw [← Finset.sum_range_add_sum_Ico _ (show n + 1 ≤ m + 1 by omega)]
    ring
  rw [hsplit]
  have h1r : 0 < 1 - |r| := by linarith
  calc |∑ k ∈ Finset.Ico (n + 1) (m + 1), r ^ k|
      ≤ ∑ k ∈ Finset.Ico (n + 1) (m + 1), |r| ^ k := by
        refine (Finset.abs_sum_le_sum_abs _ _).trans (le_of_eq ?_)
        simp [abs_pow]
    _ = |r| ^ (n + 1) * ∑ j ∈ Finset.range (m - n), |r| ^ j := by
        rw [Finset.sum_Ico_eq_sum_range, Finset.mul_sum]
        rw [show m + 1 - (n + 1) = m - n by omega]
        refine Finset.sum_congr rfl (fun j _ => by rw [pow_add])
    _ ≤ |r| ^ (n + 1) * (1 / (1 - |r|)) := by
        gcongr
        have hr1 : |r| ≠ 1 := by intro h1; linarith
        rw [geom_sum_eq hr1]
        rw [show (|r| ^ (m - n) - 1) / (|r| - 1) = (1 - |r| ^ (m - n)) / (1 - |r|) by
          rw [← neg_div_neg_eq]; ring_nf]
        rw [div_le_div_iff_of_pos_right h1r]
        have : 0 ≤ |r| ^ (m - n) := by positivity
        linarith
    _ = |r| ^ (n + 1) / (1 - |r|) := by ring

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.geom_tail (0) ((by norm_num)) (0) (1) ((by norm_num)) := by
  simp only [Finset.sum_range_succ, Finset.sum_range_zero]
  norm_num [abs_le]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.geom_tail (1 / 2) ((by norm_num)) (1) (3) ((by norm_num)) := by
  simp only [Finset.sum_range_succ, Finset.sum_range_zero]
  norm_num [abs_le]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.geom_tail (-1 / 3) ((by norm_num)) (2) (5) ((by norm_num)) := by
  simp only [Finset.sum_range_succ, Finset.sum_range_zero]
  norm_num [abs_le]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.geom_tail (9 / 10) ((by norm_num)) (0) (4) ((by norm_num)) := by
  simp only [Finset.sum_range_succ, Finset.sum_range_zero]
  norm_num [abs_le]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Sequences.geom_tail
-- test: ra-geom-tail-test.sage

/-! ## thm-monotone-convergence -/

/-- An increasing sequence bounded above converges to the supremum of its values. -/
theorem monotone_convergence (a : ℕ → ℝ) (h : Monotone a) (hb : BddAbove (range a)) :
    Tendsto a atTop (𝓝 (⨆ n, a n)) :=
  tendsto_atTop_ciSup h hb

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Sequences.monotone_convergence
-- test: ra-monotone-convergence-test.sage

/-- A decreasing sequence bounded below converges to the infimum of its values. -/
theorem antitone_convergence (a : ℕ → ℝ) (h : Antitone a) (hb : BddBelow (range a)) :
    Tendsto a atTop (𝓝 (⨅ n, a n)) :=
  tendsto_atTop_ciInf h hb

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Sequences.antitone_convergence
-- test: ra-antitone-convergence-test.sage

/-! ## exm-monotone-reciprocal -/

/-- `1/n` is decreasing and bounded below by `0`. -/
theorem recip_decreasing (n : ℕ) : 0 < 1 / ((n : ℝ) + 2) ∧ 1 / ((n : ℝ) + 2) < 1 / ((n : ℝ) + 1) := by
  refine ⟨by positivity, ?_⟩
  apply one_div_lt_one_div_of_lt (by positivity)
  linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.recip_decreasing (0) := by
  constructor <;> norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.recip_decreasing (1) := by
  constructor <;> norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.recip_decreasing (5) := by
  constructor <;> norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.recip_decreasing (99) := by
  constructor <;> norm_num

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Sequences.recip_decreasing
-- test: ra-recip-decreasing-test.sage

/-! ## exm-monotone-sqrt2 -/

/-- Newton's recursion for `√c`: `a₀ = c`, `a_{n+1} = (a_n + c/a_n)/2`. -/
noncomputable def newton (c : ℝ) : ℕ → ℝ
  | 0 => c
  | n + 1 => (newton c n + c / newton c n) / 2

theorem newton_succ (c : ℝ) (n : ℕ) : newton c (n + 1) = (newton c n + c / newton c n) / 2 := rfl

/-- For `c ≥ 1` the Newton iterates are positive, satisfy `a_n² ≥ c` and decrease. -/
theorem newton_bounds (c : ℝ) (hc : 1 ≤ c) (n : ℕ) :
    0 < newton c n ∧ c ≤ newton c n ^ 2 ∧ newton c (n + 1) ≤ newton c n := by
  have key : ∀ n, 0 < newton c n ∧ c ≤ newton c n ^ 2 := by
    intro n
    induction n with
    | zero => exact ⟨by simp [newton]; linarith, by simp [newton]; nlinarith⟩
    | succ k ih =>
      obtain ⟨hpos, hsq⟩ := ih
      rw [newton_succ]
      refine ⟨by positivity, ?_⟩
      have e : ((newton c k + c / newton c k) / 2) ^ 2 - c = ((newton c k ^ 2 - c) / (2 * newton c k)) ^ 2 := by
        field_simp
        ring
      nlinarith [sq_nonneg ((newton c k ^ 2 - c) / (2 * newton c k))]
  obtain ⟨hpos, hsq⟩ := key n
  refine ⟨hpos, hsq, ?_⟩
  rw [newton_succ]
  have : c / newton c n ≤ newton c n := by
    rw [div_le_iff₀ hpos]; nlinarith
  linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.newton_bounds (1) ((by norm_num)) (0) := by
  norm_num [KaluzaKlein.RealAnalysis.Sequences.newton]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.newton_bounds (2) ((by norm_num)) (0) := by
  norm_num [KaluzaKlein.RealAnalysis.Sequences.newton]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.newton_bounds (2) ((by norm_num)) (2) := by
  norm_num [KaluzaKlein.RealAnalysis.Sequences.newton]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.newton_bounds (3) ((by norm_num)) (1) := by
  norm_num [KaluzaKlein.RealAnalysis.Sequences.newton]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Sequences.newton_bounds
-- test: ra-newton-bounds-test.sage

/-- Newton's recursion converges to `√c` (to `√2` from `a₀ = 2`). -/
theorem newton_limit (c : ℝ) (hc : 1 ≤ c) : Tendsto (newton c) atTop (𝓝 (√c)) := by
  have hanti : Antitone (newton c) := antitone_nat_of_succ_le (fun n => (newton_bounds c hc n).2.2)
  have hbdd : BddBelow (range (newton c)) :=
    ⟨0, by rintro _ ⟨n, rfl⟩; exact (newton_bounds c hc n).1.le⟩
  have hL := tendsto_atTop_ciInf hanti hbdd
  set L := ⨅ n, newton c n with hLdef
  have hsqrt : ∀ n, √c ≤ newton c n := fun n => by
    obtain ⟨hpos, hsq, -⟩ := newton_bounds c hc n
    calc √c ≤ √(newton c n ^ 2) := Real.sqrt_le_sqrt hsq
      _ = newton c n := Real.sqrt_sq hpos.le
  have hLge : √c ≤ L := le_ciInf hsqrt
  have hc0 : 0 < √c := Real.sqrt_pos.2 (by linarith)
  have hL0 : L ≠ 0 := by linarith
  have h1 : Tendsto (fun n => newton c (n + 1)) atTop (𝓝 L) := hL.comp (tendsto_add_atTop_nat 1)
  have h2 : Tendsto (fun n => newton c (n + 1)) atTop (𝓝 ((L + c / L) / 2)) := by
    simp_rw [newton_succ]
    exact (hL.add (tendsto_const_nhds.div hL hL0)).div_const 2
  have hfix : L = (L + c / L) / 2 := tendsto_nhds_unique h1 h2
  have hLsq : L ^ 2 = c := by
    field_simp at hfix
    nlinarith
  have : L = √c := by
    rw [← hLsq, Real.sqrt_sq (by linarith)]
  rw [← this]
  exact hL

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Sequences.newton_limit
-- test: ra-newton-limit-test.sage

/-- The first iterates from `a₀ = 2`: `2, 3/2, 17/12, 577/408` (`1.5, 1.4167, 1.41422`). -/
theorem newton_table (n : Fin 4) : newton 2 n = ![2, 3 / 2, 17 / 12, 577 / 408] n := by
  fin_cases n <;> norm_num [newton]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.newton_table (0) := by
  norm_num [KaluzaKlein.RealAnalysis.Sequences.newton]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.newton_table (1) := by
  norm_num [KaluzaKlein.RealAnalysis.Sequences.newton]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.newton_table (2) := by
  norm_num [KaluzaKlein.RealAnalysis.Sequences.newton]

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.newton_table (3) := by
  norm_num [KaluzaKlein.RealAnalysis.Sequences.newton]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Sequences.newton_table
-- test: ra-newton-table-test.sage

/-! ## exm-monotone-euler -/

/-- `(1 + 1/n)^n` is increasing. -/
theorem euler_increasing (n : ℕ) (hn : 1 ≤ n) :
    (1 + 1 / (n : ℝ)) ^ n < (1 + 1 / ((n : ℝ) + 1)) ^ (n + 1) := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  set m : ℝ := (n : ℝ) + 1 with hm
  have hm1 : 1 < m := by rw [hm]; linarith
  have hm0 : 0 < m := by linarith
  -- strict Bernoulli: (1 - 1/m²)^(n+1) > 1 - (n+1)/m² = 1 - 1/m
  have hb : 1 - 1 / m < (1 - 1 / m ^ 2) ^ (n + 1) := by
    have hs1 : -1 ≤ -(1 / m ^ 2) := by
      rw [neg_le_neg_iff, div_le_one (by positivity)]; nlinarith
    have hs2 : -(1 / m ^ 2) ≠ 0 := by
      have : 0 < 1 / m ^ 2 := by positivity
      linarith
    have hp : (1 : ℝ) < ((n + 1 : ℕ) : ℝ) := by push_cast; linarith
    have h := one_add_mul_self_lt_rpow_one_add hs1 hs2 hp
    rw [Real.rpow_natCast] at h
    have e1 : (1 : ℝ) + ((n + 1 : ℕ) : ℝ) * -(1 / m ^ 2) = 1 - 1 / m := by
      push_cast
      rw [← hm]
      field_simp
      ring
    rw [e1, ← sub_eq_add_neg] at h
    exact h
  have e1 : 1 + 1 / m = (1 - 1 / m ^ 2) * (m / n) := by
    rw [hm]; field_simp; ring
  have e2 : 1 + 1 / (n : ℝ) = m / n := by
    rw [hm]; field_simp
  have e3 : (1 - 1 / m) * (m / n) = 1 := by
    rw [hm]; field_simp; ring
  rw [e1, e2, mul_pow, pow_succ (m / n)]
  have hpos : 0 < (m / n) ^ n * (m / n) := by positivity
  have := mul_lt_mul_of_pos_right hb hpos
  calc (m / n) ^ n = (1 - 1 / m) * ((m / n) ^ n * (m / n)) := by
        rw [mul_comm ((m / n) ^ n), ← mul_assoc, e3, one_mul]
    _ < _ := this

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.euler_increasing (1) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.euler_increasing (2) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.euler_increasing (3) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.euler_increasing (10) ((by norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Sequences.euler_increasing
-- test: ra-euler-increasing-test.sage

/-- `(1 + 1/n)^n < 3`: it is at most `e < 3`. -/
theorem euler_bound (n : ℕ) (hn : 1 ≤ n) : (1 + 1 / (n : ℝ)) ^ n < 3 := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  have h1 : 1 + 1 / (n : ℝ) ≤ Real.exp (1 / n) := by
    have := Real.add_one_le_exp (1 / (n : ℝ)); linarith
  have h2 : (1 + 1 / (n : ℝ)) ^ n ≤ Real.exp (1 / n) ^ n := pow_le_pow_left₀ (by positivity) h1 n
  rw [← Real.exp_nat_mul, show (n : ℝ) * (1 / n) = 1 by field_simp] at h2
  have h3 := Real.exp_one_lt_d9
  linarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.euler_bound (1) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.euler_bound (2) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.euler_bound (10) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.euler_bound (50) ((by norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Sequences.euler_bound
-- test: ra-euler-bound-test.sage

/-! ## exr-monotone-sqrt2 -/

/-- The error of one Newton step squares: `(a + c/a)/2 - √c = (a - √c)²/(2a)`. -/
theorem newton_step_error (a c s : ℝ) (ha : a ≠ 0) (hs : s ^ 2 = c) :
    (a + c / a) / 2 - s = (a - s) ^ 2 / (2 * a) := by
  rw [← hs]
  field_simp
  ring

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.newton_step_error (1) (1) (1) ((by norm_num)) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.newton_step_error (2) (4) (2) ((by norm_num)) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.newton_step_error (3 / 2) (9 / 4) (-3 / 2) ((by norm_num)) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.newton_step_error (5) (1) (1) ((by norm_num)) ((by norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Sequences.newton_step_error
-- test: ra-newton-step-error-test.sage

/-- The fixed points of the recursion: `L = (L + 2/L)/2` exactly when `L² = 2`, so the positive limit is `√2`. -/
theorem newton_fixed_point (L : ℝ) (hL : L ≠ 0) : L = (L + 2 / L) / 2 ↔ L ^ 2 = 2 := by
  constructor
  · intro h
    field_simp at h
    nlinarith
  · intro h
    field_simp
    nlinarith

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.newton_fixed_point (1) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.newton_fixed_point (2) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.newton_fixed_point (-1) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.newton_fixed_point (3 / 2) ((by norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Sequences.newton_fixed_point
-- test: ra-newton-fixed-point-test.sage

/-- The error of the Newton iterates for `√2`: `0 ≤ a_n - √2 ≤ (a_n² - 2)/2`, below `10⁻⁴⁸` at `n = 6`. -/
theorem newton_error (n : ℕ) : 0 ≤ newton 2 n - √2 ∧ newton 2 n - √2 ≤ (newton 2 n ^ 2 - 2) / 2 := by
  obtain ⟨hpos, hsq, -⟩ := newton_bounds 2 (by norm_num) n
  have hs := Real.sq_sqrt (show (0 : ℝ) ≤ 2 by norm_num)
  have h0 : 0 ≤ √(2 : ℝ) := Real.sqrt_nonneg 2
  have hge : √2 ≤ newton 2 n := by
    calc √2 ≤ √(newton 2 n ^ 2) := Real.sqrt_le_sqrt hsq
      _ = newton 2 n := Real.sqrt_sq hpos.le
  have h1 : (1 : ℝ) < √2 := by
    rw [show (1 : ℝ) = √1 by simp]; exact Real.sqrt_lt_sqrt (by norm_num) (by norm_num)
  refine ⟨by linarith, ?_⟩
  -- a - √2 = (a² - 2)/(a + √2) ≤ (a² - 2)/2 since a + √2 ≥ 2√2 > 2
  have hsum : 2 ≤ newton 2 n + √2 := by linarith
  have e : newton 2 n ^ 2 - 2 = (newton 2 n - √2) * (newton 2 n + √2) := by nlinarith
  rw [e]
  nlinarith

attribute [veriphy_tested oracle_test 0 7] KaluzaKlein.RealAnalysis.Sequences.newton_error
-- test: ra-newton-error-test.sage

/-! ## exr-monotone-e -/

/-- The `k`-th term `(1/k!) ∏_{j < k} (1 - j/n)` of the binomial expansion of `(1 + 1/n)^n`. -/
noncomputable def eulerTerm (n k : ℕ) : ℝ := (1 / (k ! : ℝ)) * ∏ j ∈ Finset.range k, (1 - j / (n : ℝ))

/-- The binomial expansion `(1 + 1/n)^n = ∑_{k ≤ n} (1/k!) ∏_{j < k} (1 - j/n)`. -/
theorem euler_binomial (n : ℕ) (hn : 1 ≤ n) :
    (1 + 1 / (n : ℝ)) ^ n = ∑ k ∈ Finset.range (n + 1), eulerTerm n k := by
  simp only [eulerTerm]
  have hn0 : (n : ℝ) ≠ 0 := by positivity
  -- ∏ (1 - j/n) = descFactorial n k / n^k
  have hprod : ∀ k, k ≤ n → ∏ j ∈ Finset.range k, (1 - j / (n : ℝ)) = (n.descFactorial k : ℝ) / (n : ℝ) ^ k := by
    intro k
    induction k with
    | zero => intro _; simp
    | succ k ih =>
      intro hk
      rw [Finset.prod_range_succ, ih (by omega), Nat.descFactorial_succ, Nat.cast_mul,
        Nat.cast_sub (by omega), pow_succ]
      field_simp <;> ring
  rw [add_comm, add_pow]
  refine Finset.sum_congr rfl (fun k hk => ?_)
  have hkn : k ≤ n := Nat.lt_succ_iff.1 (Finset.mem_range.1 hk)
  rw [one_pow, mul_one]
  rw [hprod k hkn, Nat.descFactorial_eq_factorial_mul_choose, Nat.cast_mul, div_pow, one_pow]
  have hf : (k ! : ℝ) ≠ 0 := by positivity
  field_simp

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.euler_binomial (1) ((by norm_num)) := by
  simp only [eulerTerm, Finset.sum_range_succ, Finset.prod_range_succ, Finset.sum_range_zero, Finset.prod_range_zero, Nat.factorial]
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.euler_binomial (2) ((by norm_num)) := by
  simp only [eulerTerm, Finset.sum_range_succ, Finset.prod_range_succ, Finset.sum_range_zero, Finset.prod_range_zero, Nat.factorial]
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.euler_binomial (3) ((by norm_num)) := by
  simp only [eulerTerm, Finset.sum_range_succ, Finset.prod_range_succ, Finset.sum_range_zero, Finset.prod_range_zero, Nat.factorial]
  norm_num

example : veriphy_instance% KaluzaKlein.RealAnalysis.Sequences.euler_binomial (5) ((by norm_num)) := by
  simp only [eulerTerm, Finset.sum_range_succ, Finset.prod_range_succ, Finset.sum_range_zero, Finset.prod_range_zero, Nat.factorial]
  norm_num

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.RealAnalysis.Sequences.euler_binomial
-- test: ra-euler-binomial-test.sage

/-- `e < 3`. -/
theorem exp_one_lt_three : Real.exp 1 < 3 := by
  veriphy_certified (exact lt_trans Real.exp_one_lt_d9 (by norm_num))
  -- certificate: ra-exp-one-lt-three.sage

/-! ## exr-nested-intervals -/

/-- Nested closed intervals whose lengths tend to `0` have exactly one common point. -/
theorem nested_intervals (a b : ℕ → ℝ) (hab : ∀ n, a n ≤ b n) (ha : Monotone a) (hb : Antitone b)
    (hlen : Tendsto (fun n => b n - a n) atTop (𝓝 0)) : ∃! x, ∀ n, a n ≤ x ∧ x ≤ b n := by
  have hbdd : BddAbove (range a) := ⟨b 0, by rintro _ ⟨n, rfl⟩; exact (hab n).trans (hb (Nat.zero_le n))⟩
  have hin : ∀ n, a n ≤ ⨆ m, a m ∧ ⨆ m, a m ≤ b n := fun n =>
    ⟨le_ciSup hbdd n, ciSup_le fun m => (ha (le_max_left m n)).trans ((hab _).trans (hb (le_max_right m n)))⟩
  refine ⟨⨆ m, a m, hin, fun y hy => ?_⟩
  by_contra hne
  have hpos : 0 < |y - ⨆ m, a m| := abs_pos.2 (sub_ne_zero.2 hne)
  obtain ⟨N, hN⟩ := (Metric.tendsto_atTop.1 hlen) _ hpos
  have h1 := hN N le_rfl
  rw [Real.dist_eq, sub_zero] at h1
  have h2 := hy N
  have h3 := hin N
  have : |y - ⨆ m, a m| ≤ b N - a N := by
    rw [abs_le]; constructor <;> linarith
  have : b N - a N ≤ |b N - a N| := le_abs_self _
  linarith

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Sequences.nested_intervals
-- test: ra-nested-intervals-test.sage

/-! ## prp-sequence-limits -/

/-- `n^(1/n) → 1` by the binomial estimate: `h_n = n^(1/n) - 1 ≥ 0` and `h_n² ≤ 2/(n - 1)`. -/
theorem nroot_bound (n : ℕ) (hn : 2 ≤ n) :
    0 ≤ (n : ℝ) ^ ((n : ℝ)⁻¹) - 1 ∧ ((n : ℝ) ^ ((n : ℝ)⁻¹) - 1) ^ 2 ≤ 2 / ((n : ℝ) - 1) := by
  have hn1 : (1 : ℝ) < n := by exact_mod_cast hn
  have hn0 : (0 : ℝ) < n := by linarith
  set h := (n : ℝ) ^ ((n : ℝ)⁻¹) - 1 with hh
  have hge : 1 ≤ (n : ℝ) ^ ((n : ℝ)⁻¹) := Real.one_le_rpow hn1.le (by positivity)
  have h0 : 0 ≤ h := by rw [hh]; linarith
  refine ⟨h0, ?_⟩
  have hpow : (1 + h) ^ n = n := by
    rw [hh, add_sub_cancel, ← Real.rpow_natCast, ← Real.rpow_mul hn0.le, inv_mul_cancel₀ hn0.ne',
      Real.rpow_one]
  -- (1 + h)^n ≥ C(n,2) h²
  have hbin : ((n.choose 2 : ℕ) : ℝ) * h ^ 2 ≤ (1 + h) ^ n := by
    rw [add_comm, add_pow]
    have hmem : 2 ∈ Finset.range (n + 1) := Finset.mem_range.2 (by omega)
    refine le_trans (le_of_eq ?_) (Finset.single_le_sum (f := fun k => h ^ k * 1 ^ (n - k) * (n.choose k : ℝ))
      (fun k _ => by positivity) hmem)
    simp only [one_pow, mul_one]
    ring
  rw [hpow, Nat.cast_choose_two] at hbin
  have hn1' : (0 : ℝ) < n - 1 := by linarith
  rw [le_div_iff₀ hn1']
  nlinarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Sequences.nroot_bound
-- test: ra-nroot-bound-test.sage

/-- `H_n - log n → γ` with `0 < H_n - log n - γ < 1/n`; at `n = 10⁴` the difference is within `10⁻⁴` of `γ`. -/
theorem harmonic_error (n : ℕ) (hn : 1 ≤ n) :
    0 < (harmonic n : ℝ) - Real.log n - Real.eulerMascheroniConstant ∧
      (harmonic n : ℝ) - Real.log n - Real.eulerMascheroniConstant < 1 / n := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  have hne : n ≠ 0 := by omega
  have h1 := Real.eulerMascheroniConstant_lt_eulerMascheroniSeq' n
  have h2 := Real.eulerMascheroniSeq_lt_eulerMascheroniConstant n
  simp only [Real.eulerMascheroniSeq', hne, if_false] at h1
  simp only [Real.eulerMascheroniSeq] at h2
  refine ⟨by linarith, ?_⟩
  have hlog : Real.log ((n : ℝ) + 1) - Real.log n ≤ 1 / n := by
    rw [← Real.log_div (by positivity) hn0.ne']
    have := Real.log_le_sub_one_of_pos (show 0 < ((n : ℝ) + 1) / n by positivity)
    rw [show ((n : ℝ) + 1) / n - 1 = 1 / n by field_simp <;> ring] at this
    exact this
  have hlt : Real.log ((n : ℝ) + 1) - Real.log n < 1 / n := by
    rw [← Real.log_div (by positivity) hn0.ne']
    have := Real.log_lt_sub_one_of_pos (show 0 < ((n : ℝ) + 1) / n by positivity)
      (by rw [Ne, div_eq_one_iff_eq hn0.ne']; linarith)
    rw [show ((n : ℝ) + 1) / n - 1 = 1 / n by field_simp <;> ring] at this
    exact this
  linarith

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Sequences.harmonic_error
-- test: ra-harmonic-error-test.sage

/-! ## exm-limits-sage -/

/-- `n log(1 + x/n) → x`, the step Sage uses for `(1 + 1/n)^n = exp(n log(1 + 1/n)) → e`. -/
theorem n_log_limit (x : ℝ) : Tendsto (fun n : ℕ => (n : ℝ) * Real.log (1 + x / n)) atTop (𝓝 x) := by
  apply Real.tendsto_nat_mul_log_one_add_of_tendsto
  apply tendsto_const_nhds.congr'
  filter_upwards [eventually_ne_atTop 0] with n hn
  have : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  field_simp

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Sequences.n_log_limit
-- test: ra-n-log-limit-test.sage

/-! ## exm-limits-euler-constant -/

/-- `H_n - log n` is decreasing and positive (bounded below by `γ > 0`). -/
theorem harmonic_sub_log_decreasing (n : ℕ) (hn : 1 ≤ n) :
    (harmonic (n + 1) : ℝ) - Real.log ((n + 1 : ℕ) : ℝ) < (harmonic n : ℝ) - Real.log n ∧
      0 < (harmonic n : ℝ) - Real.log n := by
  have hne : n ≠ 0 := by omega
  have h := Real.strictAnti_eulerMascheroniSeq' (Nat.lt_succ_self n)
  simp only [Real.eulerMascheroniSeq', hne, Nat.succ_ne_zero, if_false] at h
  have h1 := Real.eulerMascheroniConstant_lt_eulerMascheroniSeq' n
  simp only [Real.eulerMascheroniSeq', hne, if_false] at h1
  have h2 := Real.one_half_lt_eulerMascheroniConstant
  exact ⟨h, by linarith⟩

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Sequences.harmonic_sub_log_decreasing
-- test: ra-harmonic-sub-log-decreasing-test.sage

/-! ## exr-limit-nroot -/

/-- `a^(1/n) → 1` for every `a > 0`. -/
theorem rpow_inv_limit (a : ℝ) (ha : 0 < a) : Tendsto (fun n : ℕ => a ^ ((n : ℝ)⁻¹)) atTop (𝓝 1) := by
  have h0 : Tendsto (fun n : ℕ => Real.log a / n) atTop (𝓝 0) :=
    tendsto_const_nhds.div_atTop tendsto_natCast_atTop_atTop
  have h := (Real.continuous_exp.tendsto 0).comp h0
  rw [Real.exp_zero] at h
  apply h.congr'
  filter_upwards with n
  simp only [Function.comp_apply]
  rw [Real.rpow_def_of_pos ha, div_eq_mul_inv]

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Sequences.rpow_inv_limit
-- test: ra-rpow-inv-limit-test.sage

/-! ## exr-limit-squeeze -/

/-- The squeeze theorem. -/
theorem squeeze (a b c : ℕ → ℝ) (L : ℝ) (hab : ∀ n, a n ≤ b n) (hbc : ∀ n, b n ≤ c n)
    (ha : Tendsto a atTop (𝓝 L)) (hc : Tendsto c atTop (𝓝 L)) : Tendsto b atTop (𝓝 L) :=
  tendsto_of_tendsto_of_tendsto_of_le_of_le ha hc hab hbc

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.RealAnalysis.Sequences.squeeze
-- test: ra-squeeze-test.sage

/-- `x cos x ≤ sin x ≤ x` on `[0, 1]`, hence `cos x ≤ sin x / x ≤ 1` for `0 < x ≤ 1`. -/
theorem sin_div_bounds (x : ℝ) (hx0 : 0 < x) (hx1 : x ≤ 1) :
    Real.cos x ≤ Real.sin x / x ∧ Real.sin x / x ≤ 1 := by
  have hpi : x < π / 2 := by linarith [Real.pi_gt_three]
  have hcos : 0 < Real.cos x := Real.cos_pos_of_mem_Ioo ⟨by linarith, hpi⟩
  have htan := Real.le_tan hx0.le hpi
  rw [Real.tan_eq_sin_div_cos, le_div_iff₀ hcos] at htan
  constructor
  · rw [le_div_iff₀ hx0]; linarith
  · rw [div_le_one hx0]; exact Real.sin_le hx0.le

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Sequences.sin_div_bounds
-- test: ra-sin-div-bounds-test.sage

/-- `sin(cn)/n → 0` by the squeeze `-1/n ≤ sin(cn)/n ≤ 1/n`; at `c = 1`, `sin n / n → 0`. -/
theorem sin_div_n_limit (c : ℝ) : Tendsto (fun n : ℕ => Real.sin (c * n) / n) atTop (𝓝 0) := by
  have h1 : Tendsto (fun n : ℕ => (1 : ℝ) / n) atTop (𝓝 0) := tendsto_const_div_atTop_nhds_zero_nat 1
  have h2 : Tendsto (fun n : ℕ => -((1 : ℝ) / n)) atTop (𝓝 0) := by simpa using h1.neg
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le' h2 h1
  · filter_upwards [eventually_gt_atTop 0] with n hn
    have : (0 : ℝ) < n := by exact_mod_cast hn
    rw [neg_le, ← neg_div, div_le_div_iff_of_pos_right this]
    linarith [Real.neg_one_le_sin (c * n)]
  · filter_upwards [eventually_gt_atTop 0] with n hn
    have : (0 : ℝ) < n := by exact_mod_cast hn
    rw [div_le_div_iff_of_pos_right this]
    exact Real.sin_le_one _

attribute [veriphy_tested oracle_test 0 4] KaluzaKlein.RealAnalysis.Sequences.sin_div_n_limit
-- test: ra-sin-div-n-limit-test.sage

/-! ## exr-euler-constant -/

/-- The rate of convergence to `γ` (stated, tested numerically, not proved here): `H_n - log n - γ` lies between
`1/(2n + 1)` and `1/(2n)`, so `2n (H_n - log n - γ) → 1`. -/
def HarmonicRate : Prop :=
  ∀ n : ℕ, 1 ≤ n → 1 / (2 * (n : ℝ) + 1) < (harmonic n : ℝ) - Real.log n - Real.eulerMascheroniConstant ∧
    (harmonic n : ℝ) - Real.log n - Real.eulerMascheroniConstant < 1 / (2 * (n : ℝ))

attribute [veriphy_tested oracle_test 0 5] KaluzaKlein.RealAnalysis.Sequences.HarmonicRate
-- test: ra-harmonicrate-test.sage

end KaluzaKlein.RealAnalysis.Sequences
