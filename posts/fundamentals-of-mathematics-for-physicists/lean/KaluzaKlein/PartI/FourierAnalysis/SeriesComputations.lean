import Mathlib
import Veriphy.Kit.Levels
import KaluzaKlein.PartI.FourierAnalysis.Parseval

/-! # Fourier analysis: series computations (tutorial/part-i-06-fourier-analysis.qmd) -/

namespace KaluzaKlein.FourierAnalysis.SeriesComputations

open Real Filter Topology

/-! ## prp-fourier-series-values -/

theorem cos_two_cos_three_orthogonal : ∫ x in (0 : ℝ)..(2 * π), Real.cos (2 * x) * Real.cos (3 * x) = 0 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (2 * π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (((Real.cos x_veriphy) ^ 4) * (Real.sin x_veriphy) / 2 -
                      (((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 3)) +
                    (((Real.sin x_veriphy) ^ 5) / 10) +
                  ((Real.sin x_veriphy) / 2) :
                ℝ))
            ((fun x => cos (2 * x) * cos (3 * x)) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        have hmc2 : Real.cos (2 * x_veriphy) = ((Real.cos x_veriphy) ^ 2 - ((Real.sin x_veriphy) ^ 2) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.cos_add]; ring
        have hms2 : Real.sin (2 * x_veriphy) = (2 * (Real.cos x_veriphy) * (Real.sin x_veriphy) : ℝ) := by
          rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.sin_add]; ring
        have hmc3 :
          Real.cos (3 * x_veriphy) =
            ((Real.cos x_veriphy) ^ 3 - (3 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 2)) : ℝ) :=
          by rw [show (3 : ℝ) * x_veriphy = 2 * x_veriphy + x_veriphy by ring, Real.cos_add, hmc2, hms2]; ring
        have hms3 :
          Real.sin (3 * x_veriphy) =
            (3 * ((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) - ((Real.sin x_veriphy) ^ 3) : ℝ) :=
          by rw [show (3 : ℝ) * x_veriphy = 2 * x_veriphy + x_veriphy by ring, Real.sin_add, hmc2, hms2]; ring
        refine
          ((((((((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 4).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                                (1 / 2 : ℝ)).add
                            (((((hasDerivAt_id' (x_veriphy : ℝ)).cos).pow 2).mul
                                  (((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 3)).const_mul
                              (-1 : ℝ))).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).sin).pow 5).const_mul (1 / 10 : ℝ))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).sin).const_mul (1 / 2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (try simp only [Pi.pow_apply, Pi.mul_apply])
          simp only [hmc2, hms2, hmc3, hms3]
          linear_combination
            ((-((Real.cos x_veriphy) ^ 3) / 2 - ((Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 2) / 2) -
                  ((Real.cos x_veriphy) / 2) :
                ℝ)) *
              Real.sin_sq_add_cos_sq x_veriphy
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (((Real.cos x_veriphy) ^ 4) * (Real.sin x_veriphy) / 2 -
                  (((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 3)) +
                (((Real.sin x_veriphy) ^ 5) / 10) +
              ((Real.sin x_veriphy) / 2) :
            ℝ))
          (f' := fun x => cos (2 * x) * cos (3 * x)) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((2 * π) : ℝ)) = 1 := by
        rw [show (((2 * π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((2 * π) : ℝ)) = 0 := by
        rw [show (((2 * π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 2
      simp only [he0, he1, he2, he3]
      all_goals (norm_num; done))
  -- certificate: fa-trig-integral.sage

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 4000 in
/-- The `L²` error of `S₉` of the square wave by direct integration over `(0, π)`, doubled. -/
theorem square_l2_s9 :
    2 * ∫ x in (0 : ℝ)..π, (1 - 4 / π * (Real.sin x + Real.sin (3 * x) / 3 + Real.sin (5 * x) / 5 +
      Real.sin (7 * x) / 7 + Real.sin (9 * x) / 9)) ^ 2 =
      2 * π - 16 / π * (1 + 1 / 9 + 1 / 25 + 1 / 49 + 1 / 81) := by
  have h : ∫ x in (0 : ℝ)..π, (1 - 4 / π * (Real.sin x + Real.sin (3 * x) / 3 + Real.sin (5 * x) / 5 +
      Real.sin (7 * x) / 7 + Real.sin (9 * x) / 9)) ^ 2 =
      π - 8 / π * (1 + 1 / 9 + 1 / 25 + 1 / 49 + 1 / 81) := by
    veriphy_certified
      ( have hF :
          ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
            HasDerivAt
              (fun x_veriphy =>
                (x_veriphy + (8 * (Real.cos (9 * x_veriphy)) / (81 * Real.pi)) +
                                              (8 * (Real.cos (7 * x_veriphy)) / (49 * Real.pi)) +
                                            (8 * (Real.cos (5 * x_veriphy)) / (25 * Real.pi)) +
                                          (8 * (Real.cos (3 * x_veriphy)) / (9 * Real.pi)) +
                                        (8 * (Real.cos x_veriphy) / Real.pi) +
                                      (939752 * x_veriphy / (99225 * (Real.pi ^ 2))) -
                                    (4 * (Real.sin (18 * x_veriphy)) / (729 * (Real.pi ^ 2))) -
                                  ((Real.sin (16 * x_veriphy)) / (63 * (Real.pi ^ 2))) -
                                (572 * (Real.sin (14 * x_veriphy)) / (15435 * (Real.pi ^ 2))) -
                              (248 * (Real.sin (12 * x_veriphy)) / (2835 * (Real.pi ^ 2))) -
                            (2252 * (Real.sin (10 * x_veriphy)) / (7875 * (Real.pi ^ 2))) -
                          (62 * (Real.sin (8 * x_veriphy)) / (315 * (Real.pi ^ 2))) -
                        (572 * (Real.sin (6 * x_veriphy)) / (2835 * (Real.pi ^ 2))) -
                      (16 * (Real.sin (4 * x_veriphy)) / (63 * (Real.pi ^ 2))) -
                    (4 * (Real.sin (2 * x_veriphy)) / (9 * (Real.pi ^ 2))) :
                  ℝ))
              ((fun x =>
                  (1 - 4 / π * (sin x + sin (3 * x) / 3 + sin (5 * x) / 5 + sin (7 * x) / 7 + sin (9 * x) / 9)) ^ 2)
                x_veriphy)
              x_veriphy :=
          by
          intro x_veriphy _
          beta_reduce
          have hmc2 : Real.cos (2 * x_veriphy) = ((Real.cos x_veriphy) ^ 2 - ((Real.sin x_veriphy) ^ 2) : ℝ) := by
            rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.cos_add]; ring
          have hms2 : Real.sin (2 * x_veriphy) = (2 * (Real.cos x_veriphy) * (Real.sin x_veriphy) : ℝ) := by
            rw [show (2 : ℝ) * x_veriphy = x_veriphy + x_veriphy by ring, Real.sin_add]; ring
          have hmc3 :
            Real.cos (3 * x_veriphy) =
              ((Real.cos x_veriphy) ^ 3 - (3 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 2)) : ℝ) :=
            by rw [show (3 : ℝ) * x_veriphy = 2 * x_veriphy + x_veriphy by ring, Real.cos_add, hmc2, hms2]; ring
          have hms3 :
            Real.sin (3 * x_veriphy) =
              (3 * ((Real.cos x_veriphy) ^ 2) * (Real.sin x_veriphy) - ((Real.sin x_veriphy) ^ 3) : ℝ) :=
            by rw [show (3 : ℝ) * x_veriphy = 2 * x_veriphy + x_veriphy by ring, Real.sin_add, hmc2, hms2]; ring
          have hmc4 :
            Real.cos (4 * x_veriphy) =
              ((Real.cos x_veriphy) ^ 4 - (6 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2)) +
                  ((Real.sin x_veriphy) ^ 4) :
                ℝ) :=
            by rw [show (4 : ℝ) * x_veriphy = 3 * x_veriphy + x_veriphy by ring, Real.cos_add, hmc3, hms3]; ring
          have hms4 :
            Real.sin (4 * x_veriphy) =
              (4 * ((Real.cos x_veriphy) ^ 3) * (Real.sin x_veriphy) -
                  (4 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 3)) :
                ℝ) :=
            by rw [show (4 : ℝ) * x_veriphy = 3 * x_veriphy + x_veriphy by ring, Real.sin_add, hmc3, hms3]; ring
          have hmc5 :
            Real.cos (5 * x_veriphy) =
              ((Real.cos x_veriphy) ^ 5 - (10 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 2)) +
                  (5 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 4)) :
                ℝ) :=
            by rw [show (5 : ℝ) * x_veriphy = 4 * x_veriphy + x_veriphy by ring, Real.cos_add, hmc4, hms4]; ring
          have hms5 :
            Real.sin (5 * x_veriphy) =
              (5 * ((Real.cos x_veriphy) ^ 4) * (Real.sin x_veriphy) -
                    (10 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 3)) +
                  ((Real.sin x_veriphy) ^ 5) :
                ℝ) :=
            by rw [show (5 : ℝ) * x_veriphy = 4 * x_veriphy + x_veriphy by ring, Real.sin_add, hmc4, hms4]; ring
          have hmc6 :
            Real.cos (6 * x_veriphy) =
              ((Real.cos x_veriphy) ^ 6 - (15 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 2)) +
                    (15 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 4)) -
                  ((Real.sin x_veriphy) ^ 6) :
                ℝ) :=
            by rw [show (6 : ℝ) * x_veriphy = 5 * x_veriphy + x_veriphy by ring, Real.cos_add, hmc5, hms5]; ring
          have hms6 :
            Real.sin (6 * x_veriphy) =
              (6 * ((Real.cos x_veriphy) ^ 5) * (Real.sin x_veriphy) -
                    (20 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 3)) +
                  (6 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 5)) :
                ℝ) :=
            by rw [show (6 : ℝ) * x_veriphy = 5 * x_veriphy + x_veriphy by ring, Real.sin_add, hmc5, hms5]; ring
          have hmc7 :
            Real.cos (7 * x_veriphy) =
              ((Real.cos x_veriphy) ^ 7 - (21 * ((Real.cos x_veriphy) ^ 5) * ((Real.sin x_veriphy) ^ 2)) +
                    (35 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 4)) -
                  (7 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 6)) :
                ℝ) :=
            by rw [show (7 : ℝ) * x_veriphy = 6 * x_veriphy + x_veriphy by ring, Real.cos_add, hmc6, hms6]; ring
          have hms7 :
            Real.sin (7 * x_veriphy) =
              (7 * ((Real.cos x_veriphy) ^ 6) * (Real.sin x_veriphy) -
                      (35 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 3)) +
                    (21 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 5)) -
                  ((Real.sin x_veriphy) ^ 7) :
                ℝ) :=
            by rw [show (7 : ℝ) * x_veriphy = 6 * x_veriphy + x_veriphy by ring, Real.sin_add, hmc6, hms6]; ring
          have hmc8 :
            Real.cos (8 * x_veriphy) =
              ((Real.cos x_veriphy) ^ 8 - (28 * ((Real.cos x_veriphy) ^ 6) * ((Real.sin x_veriphy) ^ 2)) +
                      (70 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 4)) -
                    (28 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 6)) +
                  ((Real.sin x_veriphy) ^ 8) :
                ℝ) :=
            by rw [show (8 : ℝ) * x_veriphy = 7 * x_veriphy + x_veriphy by ring, Real.cos_add, hmc7, hms7]; ring
          have hms8 :
            Real.sin (8 * x_veriphy) =
              (8 * ((Real.cos x_veriphy) ^ 7) * (Real.sin x_veriphy) -
                      (56 * ((Real.cos x_veriphy) ^ 5) * ((Real.sin x_veriphy) ^ 3)) +
                    (56 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 5)) -
                  (8 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 7)) :
                ℝ) :=
            by rw [show (8 : ℝ) * x_veriphy = 7 * x_veriphy + x_veriphy by ring, Real.sin_add, hmc7, hms7]; ring
          have hmc9 :
            Real.cos (9 * x_veriphy) =
              ((Real.cos x_veriphy) ^ 9 - (36 * ((Real.cos x_veriphy) ^ 7) * ((Real.sin x_veriphy) ^ 2)) +
                      (126 * ((Real.cos x_veriphy) ^ 5) * ((Real.sin x_veriphy) ^ 4)) -
                    (84 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 6)) +
                  (9 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 8)) :
                ℝ) :=
            by rw [show (9 : ℝ) * x_veriphy = 8 * x_veriphy + x_veriphy by ring, Real.cos_add, hmc8, hms8]; ring
          have hms9 :
            Real.sin (9 * x_veriphy) =
              (9 * ((Real.cos x_veriphy) ^ 8) * (Real.sin x_veriphy) -
                        (84 * ((Real.cos x_veriphy) ^ 6) * ((Real.sin x_veriphy) ^ 3)) +
                      (126 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 5)) -
                    (36 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 7)) +
                  ((Real.sin x_veriphy) ^ 9) :
                ℝ) :=
            by rw [show (9 : ℝ) * x_veriphy = 8 * x_veriphy + x_veriphy by ring, Real.sin_add, hmc8, hms8]; ring
          have hmc10 :
            Real.cos (10 * x_veriphy) =
              ((Real.cos x_veriphy) ^ 10 - (45 * ((Real.cos x_veriphy) ^ 8) * ((Real.sin x_veriphy) ^ 2)) +
                        (210 * ((Real.cos x_veriphy) ^ 6) * ((Real.sin x_veriphy) ^ 4)) -
                      (210 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 6)) +
                    (45 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 8)) -
                  ((Real.sin x_veriphy) ^ 10) :
                ℝ) :=
            by rw [show (10 : ℝ) * x_veriphy = 9 * x_veriphy + x_veriphy by ring, Real.cos_add, hmc9, hms9]; ring
          have hms10 :
            Real.sin (10 * x_veriphy) =
              (10 * ((Real.cos x_veriphy) ^ 9) * (Real.sin x_veriphy) -
                        (120 * ((Real.cos x_veriphy) ^ 7) * ((Real.sin x_veriphy) ^ 3)) +
                      (252 * ((Real.cos x_veriphy) ^ 5) * ((Real.sin x_veriphy) ^ 5)) -
                    (120 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 7)) +
                  (10 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 9)) :
                ℝ) :=
            by rw [show (10 : ℝ) * x_veriphy = 9 * x_veriphy + x_veriphy by ring, Real.sin_add, hmc9, hms9]; ring
          have hmc11 :
            Real.cos (11 * x_veriphy) =
              ((Real.cos x_veriphy) ^ 11 - (55 * ((Real.cos x_veriphy) ^ 9) * ((Real.sin x_veriphy) ^ 2)) +
                        (330 * ((Real.cos x_veriphy) ^ 7) * ((Real.sin x_veriphy) ^ 4)) -
                      (462 * ((Real.cos x_veriphy) ^ 5) * ((Real.sin x_veriphy) ^ 6)) +
                    (165 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 8)) -
                  (11 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 10)) :
                ℝ) :=
            by rw [show (11 : ℝ) * x_veriphy = 10 * x_veriphy + x_veriphy by ring, Real.cos_add, hmc10, hms10]; ring
          have hms11 :
            Real.sin (11 * x_veriphy) =
              (11 * ((Real.cos x_veriphy) ^ 10) * (Real.sin x_veriphy) -
                          (165 * ((Real.cos x_veriphy) ^ 8) * ((Real.sin x_veriphy) ^ 3)) +
                        (462 * ((Real.cos x_veriphy) ^ 6) * ((Real.sin x_veriphy) ^ 5)) -
                      (330 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 7)) +
                    (55 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 9)) -
                  ((Real.sin x_veriphy) ^ 11) :
                ℝ) :=
            by rw [show (11 : ℝ) * x_veriphy = 10 * x_veriphy + x_veriphy by ring, Real.sin_add, hmc10, hms10]; ring
          have hmc12 :
            Real.cos (12 * x_veriphy) =
              ((Real.cos x_veriphy) ^ 12 - (66 * ((Real.cos x_veriphy) ^ 10) * ((Real.sin x_veriphy) ^ 2)) +
                          (495 * ((Real.cos x_veriphy) ^ 8) * ((Real.sin x_veriphy) ^ 4)) -
                        (924 * ((Real.cos x_veriphy) ^ 6) * ((Real.sin x_veriphy) ^ 6)) +
                      (495 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 8)) -
                    (66 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 10)) +
                  ((Real.sin x_veriphy) ^ 12) :
                ℝ) :=
            by rw [show (12 : ℝ) * x_veriphy = 11 * x_veriphy + x_veriphy by ring, Real.cos_add, hmc11, hms11]; ring
          have hms12 :
            Real.sin (12 * x_veriphy) =
              (12 * ((Real.cos x_veriphy) ^ 11) * (Real.sin x_veriphy) -
                          (220 * ((Real.cos x_veriphy) ^ 9) * ((Real.sin x_veriphy) ^ 3)) +
                        (792 * ((Real.cos x_veriphy) ^ 7) * ((Real.sin x_veriphy) ^ 5)) -
                      (792 * ((Real.cos x_veriphy) ^ 5) * ((Real.sin x_veriphy) ^ 7)) +
                    (220 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 9)) -
                  (12 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 11)) :
                ℝ) :=
            by rw [show (12 : ℝ) * x_veriphy = 11 * x_veriphy + x_veriphy by ring, Real.sin_add, hmc11, hms11]; ring
          have hmc13 :
            Real.cos (13 * x_veriphy) =
              ((Real.cos x_veriphy) ^ 13 - (78 * ((Real.cos x_veriphy) ^ 11) * ((Real.sin x_veriphy) ^ 2)) +
                          (715 * ((Real.cos x_veriphy) ^ 9) * ((Real.sin x_veriphy) ^ 4)) -
                        (1716 * ((Real.cos x_veriphy) ^ 7) * ((Real.sin x_veriphy) ^ 6)) +
                      (1287 * ((Real.cos x_veriphy) ^ 5) * ((Real.sin x_veriphy) ^ 8)) -
                    (286 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 10)) +
                  (13 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 12)) :
                ℝ) :=
            by rw [show (13 : ℝ) * x_veriphy = 12 * x_veriphy + x_veriphy by ring, Real.cos_add, hmc12, hms12]; ring
          have hms13 :
            Real.sin (13 * x_veriphy) =
              (13 * ((Real.cos x_veriphy) ^ 12) * (Real.sin x_veriphy) -
                            (286 * ((Real.cos x_veriphy) ^ 10) * ((Real.sin x_veriphy) ^ 3)) +
                          (1287 * ((Real.cos x_veriphy) ^ 8) * ((Real.sin x_veriphy) ^ 5)) -
                        (1716 * ((Real.cos x_veriphy) ^ 6) * ((Real.sin x_veriphy) ^ 7)) +
                      (715 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 9)) -
                    (78 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 11)) +
                  ((Real.sin x_veriphy) ^ 13) :
                ℝ) :=
            by rw [show (13 : ℝ) * x_veriphy = 12 * x_veriphy + x_veriphy by ring, Real.sin_add, hmc12, hms12]; ring
          have hmc14 :
            Real.cos (14 * x_veriphy) =
              ((Real.cos x_veriphy) ^ 14 - (91 * ((Real.cos x_veriphy) ^ 12) * ((Real.sin x_veriphy) ^ 2)) +
                            (1001 * ((Real.cos x_veriphy) ^ 10) * ((Real.sin x_veriphy) ^ 4)) -
                          (3003 * ((Real.cos x_veriphy) ^ 8) * ((Real.sin x_veriphy) ^ 6)) +
                        (3003 * ((Real.cos x_veriphy) ^ 6) * ((Real.sin x_veriphy) ^ 8)) -
                      (1001 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 10)) +
                    (91 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 12)) -
                  ((Real.sin x_veriphy) ^ 14) :
                ℝ) :=
            by rw [show (14 : ℝ) * x_veriphy = 13 * x_veriphy + x_veriphy by ring, Real.cos_add, hmc13, hms13]; ring
          have hms14 :
            Real.sin (14 * x_veriphy) =
              (14 * ((Real.cos x_veriphy) ^ 13) * (Real.sin x_veriphy) -
                            (364 * ((Real.cos x_veriphy) ^ 11) * ((Real.sin x_veriphy) ^ 3)) +
                          (2002 * ((Real.cos x_veriphy) ^ 9) * ((Real.sin x_veriphy) ^ 5)) -
                        (3432 * ((Real.cos x_veriphy) ^ 7) * ((Real.sin x_veriphy) ^ 7)) +
                      (2002 * ((Real.cos x_veriphy) ^ 5) * ((Real.sin x_veriphy) ^ 9)) -
                    (364 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 11)) +
                  (14 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 13)) :
                ℝ) :=
            by rw [show (14 : ℝ) * x_veriphy = 13 * x_veriphy + x_veriphy by ring, Real.sin_add, hmc13, hms13]; ring
          have hmc15 :
            Real.cos (15 * x_veriphy) =
              ((Real.cos x_veriphy) ^ 15 - (105 * ((Real.cos x_veriphy) ^ 13) * ((Real.sin x_veriphy) ^ 2)) +
                            (1365 * ((Real.cos x_veriphy) ^ 11) * ((Real.sin x_veriphy) ^ 4)) -
                          (5005 * ((Real.cos x_veriphy) ^ 9) * ((Real.sin x_veriphy) ^ 6)) +
                        (6435 * ((Real.cos x_veriphy) ^ 7) * ((Real.sin x_veriphy) ^ 8)) -
                      (3003 * ((Real.cos x_veriphy) ^ 5) * ((Real.sin x_veriphy) ^ 10)) +
                    (455 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 12)) -
                  (15 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 14)) :
                ℝ) :=
            by rw [show (15 : ℝ) * x_veriphy = 14 * x_veriphy + x_veriphy by ring, Real.cos_add, hmc14, hms14]; ring
          have hms15 :
            Real.sin (15 * x_veriphy) =
              (15 * ((Real.cos x_veriphy) ^ 14) * (Real.sin x_veriphy) -
                              (455 * ((Real.cos x_veriphy) ^ 12) * ((Real.sin x_veriphy) ^ 3)) +
                            (3003 * ((Real.cos x_veriphy) ^ 10) * ((Real.sin x_veriphy) ^ 5)) -
                          (6435 * ((Real.cos x_veriphy) ^ 8) * ((Real.sin x_veriphy) ^ 7)) +
                        (5005 * ((Real.cos x_veriphy) ^ 6) * ((Real.sin x_veriphy) ^ 9)) -
                      (1365 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 11)) +
                    (105 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 13)) -
                  ((Real.sin x_veriphy) ^ 15) :
                ℝ) :=
            by rw [show (15 : ℝ) * x_veriphy = 14 * x_veriphy + x_veriphy by ring, Real.sin_add, hmc14, hms14]; ring
          have hmc16 :
            Real.cos (16 * x_veriphy) =
              ((Real.cos x_veriphy) ^ 16 - (120 * ((Real.cos x_veriphy) ^ 14) * ((Real.sin x_veriphy) ^ 2)) +
                              (1820 * ((Real.cos x_veriphy) ^ 12) * ((Real.sin x_veriphy) ^ 4)) -
                            (8008 * ((Real.cos x_veriphy) ^ 10) * ((Real.sin x_veriphy) ^ 6)) +
                          (12870 * ((Real.cos x_veriphy) ^ 8) * ((Real.sin x_veriphy) ^ 8)) -
                        (8008 * ((Real.cos x_veriphy) ^ 6) * ((Real.sin x_veriphy) ^ 10)) +
                      (1820 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 12)) -
                    (120 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 14)) +
                  ((Real.sin x_veriphy) ^ 16) :
                ℝ) :=
            by rw [show (16 : ℝ) * x_veriphy = 15 * x_veriphy + x_veriphy by ring, Real.cos_add, hmc15, hms15]; ring
          have hms16 :
            Real.sin (16 * x_veriphy) =
              (16 * ((Real.cos x_veriphy) ^ 15) * (Real.sin x_veriphy) -
                              (560 * ((Real.cos x_veriphy) ^ 13) * ((Real.sin x_veriphy) ^ 3)) +
                            (4368 * ((Real.cos x_veriphy) ^ 11) * ((Real.sin x_veriphy) ^ 5)) -
                          (11440 * ((Real.cos x_veriphy) ^ 9) * ((Real.sin x_veriphy) ^ 7)) +
                        (11440 * ((Real.cos x_veriphy) ^ 7) * ((Real.sin x_veriphy) ^ 9)) -
                      (4368 * ((Real.cos x_veriphy) ^ 5) * ((Real.sin x_veriphy) ^ 11)) +
                    (560 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 13)) -
                  (16 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 15)) :
                ℝ) :=
            by rw [show (16 : ℝ) * x_veriphy = 15 * x_veriphy + x_veriphy by ring, Real.sin_add, hmc15, hms15]; ring
          have hmc17 :
            Real.cos (17 * x_veriphy) =
              ((Real.cos x_veriphy) ^ 17 - (136 * ((Real.cos x_veriphy) ^ 15) * ((Real.sin x_veriphy) ^ 2)) +
                              (2380 * ((Real.cos x_veriphy) ^ 13) * ((Real.sin x_veriphy) ^ 4)) -
                            (12376 * ((Real.cos x_veriphy) ^ 11) * ((Real.sin x_veriphy) ^ 6)) +
                          (24310 * ((Real.cos x_veriphy) ^ 9) * ((Real.sin x_veriphy) ^ 8)) -
                        (19448 * ((Real.cos x_veriphy) ^ 7) * ((Real.sin x_veriphy) ^ 10)) +
                      (6188 * ((Real.cos x_veriphy) ^ 5) * ((Real.sin x_veriphy) ^ 12)) -
                    (680 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 14)) +
                  (17 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 16)) :
                ℝ) :=
            by rw [show (17 : ℝ) * x_veriphy = 16 * x_veriphy + x_veriphy by ring, Real.cos_add, hmc16, hms16]; ring
          have hms17 :
            Real.sin (17 * x_veriphy) =
              (17 * ((Real.cos x_veriphy) ^ 16) * (Real.sin x_veriphy) -
                                (680 * ((Real.cos x_veriphy) ^ 14) * ((Real.sin x_veriphy) ^ 3)) +
                              (6188 * ((Real.cos x_veriphy) ^ 12) * ((Real.sin x_veriphy) ^ 5)) -
                            (19448 * ((Real.cos x_veriphy) ^ 10) * ((Real.sin x_veriphy) ^ 7)) +
                          (24310 * ((Real.cos x_veriphy) ^ 8) * ((Real.sin x_veriphy) ^ 9)) -
                        (12376 * ((Real.cos x_veriphy) ^ 6) * ((Real.sin x_veriphy) ^ 11)) +
                      (2380 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 13)) -
                    (136 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 15)) +
                  ((Real.sin x_veriphy) ^ 17) :
                ℝ) :=
            by rw [show (17 : ℝ) * x_veriphy = 16 * x_veriphy + x_veriphy by ring, Real.sin_add, hmc16, hms16]; ring
          have hmc18 :
            Real.cos (18 * x_veriphy) =
              ((Real.cos x_veriphy) ^ 18 - (153 * ((Real.cos x_veriphy) ^ 16) * ((Real.sin x_veriphy) ^ 2)) +
                                (3060 * ((Real.cos x_veriphy) ^ 14) * ((Real.sin x_veriphy) ^ 4)) -
                              (18564 * ((Real.cos x_veriphy) ^ 12) * ((Real.sin x_veriphy) ^ 6)) +
                            (43758 * ((Real.cos x_veriphy) ^ 10) * ((Real.sin x_veriphy) ^ 8)) -
                          (43758 * ((Real.cos x_veriphy) ^ 8) * ((Real.sin x_veriphy) ^ 10)) +
                        (18564 * ((Real.cos x_veriphy) ^ 6) * ((Real.sin x_veriphy) ^ 12)) -
                      (3060 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 14)) +
                    (153 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 16)) -
                  ((Real.sin x_veriphy) ^ 18) :
                ℝ) :=
            by rw [show (18 : ℝ) * x_veriphy = 17 * x_veriphy + x_veriphy by ring, Real.cos_add, hmc17, hms17]; ring
          have hms18 :
            Real.sin (18 * x_veriphy) =
              (18 * ((Real.cos x_veriphy) ^ 17) * (Real.sin x_veriphy) -
                                (816 * ((Real.cos x_veriphy) ^ 15) * ((Real.sin x_veriphy) ^ 3)) +
                              (8568 * ((Real.cos x_veriphy) ^ 13) * ((Real.sin x_veriphy) ^ 5)) -
                            (31824 * ((Real.cos x_veriphy) ^ 11) * ((Real.sin x_veriphy) ^ 7)) +
                          (48620 * ((Real.cos x_veriphy) ^ 9) * ((Real.sin x_veriphy) ^ 9)) -
                        (31824 * ((Real.cos x_veriphy) ^ 7) * ((Real.sin x_veriphy) ^ 11)) +
                      (8568 * ((Real.cos x_veriphy) ^ 5) * ((Real.sin x_veriphy) ^ 13)) -
                    (816 * ((Real.cos x_veriphy) ^ 3) * ((Real.sin x_veriphy) ^ 15)) +
                  (18 * (Real.cos x_veriphy) * ((Real.sin x_veriphy) ^ 17)) :
                ℝ) :=
            by rw [show (18 : ℝ) * x_veriphy = 17 * x_veriphy + x_veriphy by ring, Real.sin_add, hmc17, hms17]; ring
          refine
            ((((((((((((((((((hasDerivAt_id' (x_veriphy : ℝ)).add
                                                                              ((((hasDerivAt_id'
                                                                                          (x_veriphy : ℝ)).const_mul
                                                                                      (9 : ℝ)).cos).const_mul
                                                                                (8 / (81 * Real.pi) : ℝ))).add
                                                                          ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul
                                                                                  (7 : ℝ)).cos).const_mul
                                                                            (8 / (49 * Real.pi) : ℝ))).add
                                                                      ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul
                                                                              (5 : ℝ)).cos).const_mul
                                                                        (8 / (25 * Real.pi) : ℝ))).add
                                                                  ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul
                                                                          (3 : ℝ)).cos).const_mul
                                                                    (8 / (9 * Real.pi) : ℝ))).add
                                                              (((hasDerivAt_id' (x_veriphy : ℝ)).cos).const_mul
                                                                (8 / Real.pi : ℝ))).add
                                                          ((hasDerivAt_id' (x_veriphy : ℝ)).const_mul
                                                            (939752 / (99225 * (Real.pi ^ 2)) : ℝ))).add
                                                      ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul
                                                              (18 : ℝ)).sin).const_mul
                                                        (-4 / (729 * (Real.pi ^ 2)) : ℝ))).add
                                                  ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (16 : ℝ)).sin).const_mul
                                                    (-1 / (63 * (Real.pi ^ 2)) : ℝ))).add
                                              ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (14 : ℝ)).sin).const_mul
                                                (-572 / (15435 * (Real.pi ^ 2)) : ℝ))).add
                                          ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (12 : ℝ)).sin).const_mul
                                            (-248 / (2835 * (Real.pi ^ 2)) : ℝ))).add
                                      ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (10 : ℝ)).sin).const_mul
                                        (-2252 / (7875 * (Real.pi ^ 2)) : ℝ))).add
                                  ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (8 : ℝ)).sin).const_mul
                                    (-62 / (315 * (Real.pi ^ 2)) : ℝ))).add
                              ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (6 : ℝ)).sin).const_mul
                                (-572 / (2835 * (Real.pi ^ 2)) : ℝ))).add
                          ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (4 : ℝ)).sin).const_mul
                            (-16 / (63 * (Real.pi ^ 2)) : ℝ))).add
                      ((((hasDerivAt_id' (x_veriphy : ℝ)).const_mul (2 : ℝ)).sin).const_mul
                        (-4 / (9 * (Real.pi ^ 2)) : ℝ)))).congr_of_eventuallyEq
                  (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
              ?_
          · beta_reduce; (simp; ring1)
          · (try simp only [Pi.pow_apply, Pi.mul_apply])
            simp only [hmc2, hms2, hmc3, hms3, hmc4, hms4, hmc5, hms5, hmc6, hms6, hmc7, hms7, hmc8, hms8, hmc9, hms9,
              hmc10, hms10, hmc11, hms11, hmc12, hms12, hmc13, hms13, hmc14, hms14, hmc15, hms15, hmc16, hms16, hmc17,
              hms17, hmc18, hms18]
            have hπ : Real.pi ≠ 0 := Real.pi_ne_zero
            linear_combination (norm := ring1)
              ((-8 * ((Real.cos x_veriphy) ^ 16) / (81 * (Real.pi ^ 2)) -
                                                                                                          (64 *
                                                                                                                ((Real.cos
                                                                                                                    x_veriphy) ^
                                                                                                                  14) *
                                                                                                              ((Real.sin
                                                                                                                  x_veriphy) ^
                                                                                                                2) /
                                                                                                            (81 *
                                                                                                              (Real.pi ^
                                                                                                                2))) -
                                                                                                        (224 *
                                                                                                              ((Real.cos
                                                                                                                  x_veriphy) ^
                                                                                                                12) *
                                                                                                            ((Real.sin
                                                                                                                x_veriphy) ^
                                                                                                              4) /
                                                                                                          (81 *
                                                                                                            (Real.pi ^
                                                                                                              2))) -
                                                                                                      (448 *
                                                                                                            ((Real.cos
                                                                                                                x_veriphy) ^
                                                                                                              10) *
                                                                                                          ((Real.sin
                                                                                                              x_veriphy) ^
                                                                                                            6) /
                                                                                                        (81 *
                                                                                                          (Real.pi ^
                                                                                                            2))) -
                                                                                                    (560 *
                                                                                                          ((Real.cos
                                                                                                              x_veriphy) ^
                                                                                                            8) *
                                                                                                        ((Real.sin
                                                                                                            x_veriphy) ^
                                                                                                          8) /
                                                                                                      (81 *
                                                                                                        (Real.pi ^ 2))) -
                                                                                                  (448 *
                                                                                                        ((Real.cos
                                                                                                            x_veriphy) ^
                                                                                                          6) *
                                                                                                      ((Real.sin
                                                                                                          x_veriphy) ^
                                                                                                        10) /
                                                                                                    (81 *
                                                                                                      (Real.pi ^ 2))) -
                                                                                                (224 *
                                                                                                      ((Real.cos
                                                                                                          x_veriphy) ^
                                                                                                        4) *
                                                                                                    ((Real.sin
                                                                                                        x_veriphy) ^
                                                                                                      12) /
                                                                                                  (81 * (Real.pi ^ 2))) -
                                                                                              (64 *
                                                                                                    ((Real.cos
                                                                                                        x_veriphy) ^
                                                                                                      2) *
                                                                                                  ((Real.sin x_veriphy) ^
                                                                                                    14) /
                                                                                                (81 * (Real.pi ^ 2))) -
                                                                                            (8 *
                                                                                                ((Real.sin x_veriphy) ^
                                                                                                  16) /
                                                                                              (81 * (Real.pi ^ 2))) -
                                                                                          (200 *
                                                                                              ((Real.cos x_veriphy) ^
                                                                                                14) /
                                                                                            (567 * (Real.pi ^ 2))) -
                                                                                        (1112 *
                                                                                              ((Real.cos x_veriphy) ^
                                                                                                12) *
                                                                                            ((Real.sin x_veriphy) ^ 2) /
                                                                                          (567 * (Real.pi ^ 2))) -
                                                                                      (824 * ((Real.cos x_veriphy) ^ 10) *
                                                                                          ((Real.sin x_veriphy) ^ 4) /
                                                                                        (189 * (Real.pi ^ 2))) -
                                                                                    (2680 * ((Real.cos x_veriphy) ^ 8) *
                                                                                        ((Real.sin x_veriphy) ^ 6) /
                                                                                      (567 * (Real.pi ^ 2))) -
                                                                                  (1240 * ((Real.cos x_veriphy) ^ 6) *
                                                                                      ((Real.sin x_veriphy) ^ 8) /
                                                                                    (567 * (Real.pi ^ 2))) +
                                                                                (40 * ((Real.cos x_veriphy) ^ 4) *
                                                                                    ((Real.sin x_veriphy) ^ 10) /
                                                                                  (189 * (Real.pi ^ 2))) +
                                                                              (328 * ((Real.cos x_veriphy) ^ 2) *
                                                                                  ((Real.sin x_veriphy) ^ 12) /
                                                                                (567 * (Real.pi ^ 2))) +
                                                                            (88 * ((Real.sin x_veriphy) ^ 14) /
                                                                              (567 * (Real.pi ^ 2))) -
                                                                          (17296 * ((Real.cos x_veriphy) ^ 12) /
                                                                            (19845 * (Real.pi ^ 2))) -
                                                                        (12416 * ((Real.cos x_veriphy) ^ 10) *
                                                                            ((Real.sin x_veriphy) ^ 2) /
                                                                          (6615 * (Real.pi ^ 2))) +
                                                                      (5584 * ((Real.cos x_veriphy) ^ 8) *
                                                                          ((Real.sin x_veriphy) ^ 4) /
                                                                        (6615 * (Real.pi ^ 2))) +
                                                                    (93568 * ((Real.cos x_veriphy) ^ 6) *
                                                                        ((Real.sin x_veriphy) ^ 6) /
                                                                      (19845 * (Real.pi ^ 2))) +
                                                                  (22384 * ((Real.cos x_veriphy) ^ 4) *
                                                                      ((Real.sin x_veriphy) ^ 8) /
                                                                    (6615 * (Real.pi ^ 2))) +
                                                                (1024 * ((Real.cos x_veriphy) ^ 2) *
                                                                    ((Real.sin x_veriphy) ^ 10) /
                                                                  (6615 * (Real.pi ^ 2))) -
                                                              (7216 * ((Real.sin x_veriphy) ^ 12) /
                                                                (19845 * (Real.pi ^ 2))) -
                                                            (38128 * ((Real.cos x_veriphy) ^ 10) /
                                                              (19845 * (Real.pi ^ 2))) +
                                                          (105712 * ((Real.cos x_veriphy) ^ 8) *
                                                              ((Real.sin x_veriphy) ^ 2) /
                                                            (19845 * (Real.pi ^ 2))) +
                                                        (36640 * ((Real.cos x_veriphy) ^ 6) * ((Real.sin x_veriphy) ^ 4) /
                                                          (3969 * (Real.pi ^ 2))) -
                                                      (89632 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 6) /
                                                        (19845 * (Real.pi ^ 2))) -
                                                    (115376 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 8) /
                                                      (19845 * (Real.pi ^ 2))) +
                                                  (13616 * ((Real.sin x_veriphy) ^ 10) / (19845 * (Real.pi ^ 2))) -
                                                (474392 * ((Real.cos x_veriphy) ^ 8) / (99225 * (Real.pi ^ 2))) +
                                              (5833792 * ((Real.cos x_veriphy) ^ 6) * ((Real.sin x_veriphy) ^ 2) /
                                                (99225 * (Real.pi ^ 2))) -
                                            (3861904 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 4) /
                                              (33075 * (Real.pi ^ 2))) +
                                          (4469632 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 6) /
                                            (99225 * (Real.pi ^ 2))) -
                                        (215672 * ((Real.sin x_veriphy) ^ 8) / (99225 * (Real.pi ^ 2))) -
                                      (630632 * ((Real.cos x_veriphy) ^ 6) / (99225 * (Real.pi ^ 2))) +
                                    (1496248 * ((Real.cos x_veriphy) ^ 4) * ((Real.sin x_veriphy) ^ 2) /
                                      (33075 * (Real.pi ^ 2))) -
                                  (1242152 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 4) /
                                    (33075 * (Real.pi ^ 2))) +
                                (293368 * ((Real.sin x_veriphy) ^ 6) / (99225 * (Real.pi ^ 2))) -
                              (750752 * ((Real.cos x_veriphy) ^ 4) / (99225 * (Real.pi ^ 2))) +
                            (2278496 * ((Real.cos x_veriphy) ^ 2) * ((Real.sin x_veriphy) ^ 2) /
                              (99225 * (Real.pi ^ 2))) -
                          (397952 * ((Real.sin x_veriphy) ^ 4) / (99225 * (Real.pi ^ 2))) -
                        (851552 * ((Real.cos x_veriphy) ^ 2) / (99225 * (Real.pi ^ 2))) +
                      (559648 * ((Real.sin x_veriphy) ^ 2) / (99225 * (Real.pi ^ 2))) -
                    (939752 / (99225 * (Real.pi ^ 2))) :
                  ℝ)) *
                Real.sin_sq_add_cos_sq x_veriphy
        rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
            (x_veriphy + (8 * (Real.cos (9 * x_veriphy)) / (81 * Real.pi)) +
                                          (8 * (Real.cos (7 * x_veriphy)) / (49 * Real.pi)) +
                                        (8 * (Real.cos (5 * x_veriphy)) / (25 * Real.pi)) +
                                      (8 * (Real.cos (3 * x_veriphy)) / (9 * Real.pi)) +
                                    (8 * (Real.cos x_veriphy) / Real.pi) +
                                  (939752 * x_veriphy / (99225 * (Real.pi ^ 2))) -
                                (4 * (Real.sin (18 * x_veriphy)) / (729 * (Real.pi ^ 2))) -
                              ((Real.sin (16 * x_veriphy)) / (63 * (Real.pi ^ 2))) -
                            (572 * (Real.sin (14 * x_veriphy)) / (15435 * (Real.pi ^ 2))) -
                          (248 * (Real.sin (12 * x_veriphy)) / (2835 * (Real.pi ^ 2))) -
                        (2252 * (Real.sin (10 * x_veriphy)) / (7875 * (Real.pi ^ 2))) -
                      (62 * (Real.sin (8 * x_veriphy)) / (315 * (Real.pi ^ 2))) -
                    (572 * (Real.sin (6 * x_veriphy)) / (2835 * (Real.pi ^ 2))) -
                  (16 * (Real.sin (4 * x_veriphy)) / (63 * (Real.pi ^ 2))) -
                (4 * (Real.sin (2 * x_veriphy)) / (9 * (Real.pi ^ 2))) :
              ℝ))
            (f' := fun x =>
            (1 - 4 / π * (sin x + sin (3 * x) / 3 + sin (5 * x) / 5 + sin (7 * x) / 7 + sin (9 * x) / 9)) ^ 2) hF
            (by apply Continuous.intervalIntegrable; fun_prop)]
        beta_reduce
        have he0 : Real.cos ((9 * (0) : ℝ)) = 1 := by simp
        have he1 : Real.cos ((9 * (π) : ℝ)) = -1 :=
          by
          rw [show ((9 * (π) : ℝ) : ℝ) = ((9 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]
          norm_num
        have he2 : Real.cos ((7 * (0) : ℝ)) = 1 := by simp
        have he3 : Real.cos ((7 * (π) : ℝ)) = -1 :=
          by
          rw [show ((7 * (π) : ℝ) : ℝ) = ((7 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]
          norm_num
        have he4 : Real.cos ((5 * (0) : ℝ)) = 1 := by simp
        have he5 : Real.cos ((5 * (π) : ℝ)) = -1 :=
          by
          rw [show ((5 * (π) : ℝ) : ℝ) = ((5 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]
          norm_num
        have he6 : Real.cos ((3 * (0) : ℝ)) = 1 := by simp
        have he7 : Real.cos ((3 * (π) : ℝ)) = -1 :=
          by
          rw [show ((3 * (π) : ℝ) : ℝ) = ((3 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]
          norm_num
        have he8 : Real.cos (((0) : ℝ)) = 1 := by simp
        have he9 : Real.cos (((π) : ℝ)) = -1 :=
          by
          rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]
          norm_num
        have he10 : Real.sin ((18 * (0) : ℝ)) = 0 := by simp
        have he11 : Real.sin ((18 * (π) : ℝ)) = 0 :=
          by
          rw [show ((18 * (π) : ℝ) : ℝ) = ((18 : ℕ) : ℝ) * Real.pi by push_cast; ring]
          exact Real.sin_nat_mul_pi 18
        have he12 : Real.sin ((16 * (0) : ℝ)) = 0 := by simp
        have he13 : Real.sin ((16 * (π) : ℝ)) = 0 :=
          by
          rw [show ((16 * (π) : ℝ) : ℝ) = ((16 : ℕ) : ℝ) * Real.pi by push_cast; ring]
          exact Real.sin_nat_mul_pi 16
        have he14 : Real.sin ((14 * (0) : ℝ)) = 0 := by simp
        have he15 : Real.sin ((14 * (π) : ℝ)) = 0 :=
          by
          rw [show ((14 * (π) : ℝ) : ℝ) = ((14 : ℕ) : ℝ) * Real.pi by push_cast; ring]
          exact Real.sin_nat_mul_pi 14
        have he16 : Real.sin ((12 * (0) : ℝ)) = 0 := by simp
        have he17 : Real.sin ((12 * (π) : ℝ)) = 0 :=
          by
          rw [show ((12 * (π) : ℝ) : ℝ) = ((12 : ℕ) : ℝ) * Real.pi by push_cast; ring]
          exact Real.sin_nat_mul_pi 12
        have he18 : Real.sin ((10 * (0) : ℝ)) = 0 := by simp
        have he19 : Real.sin ((10 * (π) : ℝ)) = 0 :=
          by
          rw [show ((10 * (π) : ℝ) : ℝ) = ((10 : ℕ) : ℝ) * Real.pi by push_cast; ring]
          exact Real.sin_nat_mul_pi 10
        have he20 : Real.sin ((8 * (0) : ℝ)) = 0 := by simp
        have he21 : Real.sin ((8 * (π) : ℝ)) = 0 :=
          by
          rw [show ((8 * (π) : ℝ) : ℝ) = ((8 : ℕ) : ℝ) * Real.pi by push_cast; ring]
          exact Real.sin_nat_mul_pi 8
        have he22 : Real.sin ((6 * (0) : ℝ)) = 0 := by simp
        have he23 : Real.sin ((6 * (π) : ℝ)) = 0 :=
          by
          rw [show ((6 * (π) : ℝ) : ℝ) = ((6 : ℕ) : ℝ) * Real.pi by push_cast; ring]
          exact Real.sin_nat_mul_pi 6
        have he24 : Real.sin ((4 * (0) : ℝ)) = 0 := by simp
        have he25 : Real.sin ((4 * (π) : ℝ)) = 0 :=
          by
          rw [show ((4 * (π) : ℝ) : ℝ) = ((4 : ℕ) : ℝ) * Real.pi by push_cast; ring]
          exact Real.sin_nat_mul_pi 4
        have he26 : Real.sin ((2 * (0) : ℝ)) = 0 := by simp
        have he27 : Real.sin ((2 * (π) : ℝ)) = 0 :=
          by
          rw [show ((2 * (π) : ℝ) : ℝ) = ((2 : ℕ) : ℝ) * Real.pi by push_cast; ring]
          exact Real.sin_nat_mul_pi 2
        simp only [he0, he1, he2, he3, he4, he5, he6, he7, he8, he9, he10, he11, he12, he13, he14, he15, he16, he17, he18,
          he19, he20, he21, he22, he23, he24, he25, he26, he27]
        all_goals (field_simp; ring1))
    -- certificate: fa-trig-integral-ma.sage
  rw [h]; ring

/-- The numerical value of the `S₉` defect from any enclosure `3.141592 < p < 3.141593` of `π`. -/
theorem s9_value_of_bounds (p : ℝ) (h1 : 3.141592 < p) (h2 : p < 3.141593) :
    0.2538 < 2 * p - 16 / p * (1 + 1 / 9 + 1 / 25 + 1 / 49 + 1 / 81) ∧
      2 * p - 16 / p * (1 + 1 / 9 + 1 / 25 + 1 / 49 + 1 / 81) < 0.2539 := by
  have hp : 0 < p := by linarith
  have e : 2 * p - 16 / p * (1 + 1 / 9 + 1 / 25 + 1 / 49 + 1 / 81) =
      (2 * p ^ 2 - 16 * (1 + 1 / 9 + 1 / 25 + 1 / 49 + 1 / 81)) / p := by
    first | (field_simp; done) | (field_simp; ring)
  rw [e]
  constructor
  · rw [lt_div_iff₀ hp]; nlinarith [sq_nonneg (p - 3.141592)]
  · rw [div_lt_iff₀ hp]; nlinarith [mul_pos hp (sub_pos.2 h2)]

example : veriphy_instance% KaluzaKlein.FourierAnalysis.SeriesComputations.s9_value_of_bounds (3.1415921) ((by norm_num)) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.FourierAnalysis.SeriesComputations.s9_value_of_bounds (3.1415925) ((by norm_num)) ((by norm_num)) := by
  norm_num

example : veriphy_instance% KaluzaKlein.FourierAnalysis.SeriesComputations.s9_value_of_bounds (3.1415929) ((by norm_num)) ((by norm_num)) := by
  norm_num

attribute [veriphy_tested kernel_test 3 0] KaluzaKlein.FourierAnalysis.SeriesComputations.s9_value_of_bounds
-- test: fa-s9-value-of-bounds-test.sage

/-- The value `0.2538…` of the `L²` error, from certified bounds on `π`. -/
theorem square_l2_s9_value :
    0.2538 < 2 * π - 16 / π * (1 + 1 / 9 + 1 / 25 + 1 / 49 + 1 / 81) ∧
      2 * π - 16 / π * (1 + 1 / 9 + 1 / 25 + 1 / 49 + 1 / 81) < 0.2539 :=
  s9_value_of_bounds π Real.pi_gt_d6 Real.pi_lt_d6

/-! ## exm-fs-sage-piecewise -/

/-- The two pieces of the square wave's coefficient are equal: `-∫_{-π}^0 sin n x = ∫_0^π sin n x = (1 - cos n π)/n`. -/
theorem square_pieces (n : ℕ) (hn : n ≠ 0) :
    -∫ x in (-π)..0, Real.sin (n * x) = (1 - Real.cos (n * π)) / n ∧
      ∫ x in (0 : ℝ)..π, Real.sin (n * x) = (1 - Real.cos (n * π)) / n := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  constructor
  · rw [intervalIntegral.integral_comp_mul_left (fun x => Real.sin x) hn', integral_sin, smul_eq_mul]
    simp only [mul_zero, Real.cos_zero, mul_neg, Real.cos_neg]
    field_simp; ring
  · rw [intervalIntegral.integral_comp_mul_left (fun x => Real.sin x) hn', integral_sin, smul_eq_mul]
    simp only [mul_zero, Real.cos_zero]
    field_simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.SeriesComputations.square_pieces
-- test: fa-square-pieces-test.sage

/-! ## exr-fs-triangle-wave -/

/-- The cosine coefficients of the triangle wave `π/2 - |x|`: `2 (1 - cos n π) / (π n²)`. -/
theorem triangle_an (n : ℕ) (hn : n ≠ 0) :
    2 / π * ∫ x in (0 : ℝ)..π, (π / 2 - x) * Real.cos (n * x) = 2 * (1 - Real.cos (n * π)) / (π * n ^ 2) := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  have hF : ∀ x ∈ Set.uIcc 0 π, HasDerivAt
      (fun x => (π / 2 - x) * Real.sin (n * x) / n - Real.cos (n * x) / n ^ 2) ((π / 2 - x) * Real.cos (n * x)) x := by
    intro x _
    have hs : HasDerivAt (fun x => Real.sin (n * x)) (Real.cos (n * x) * n) x := by
      simpa using ((hasDerivAt_id' x).const_mul (n : ℝ)).sin
    have hc : HasDerivAt (fun x => Real.cos (n * x)) (-Real.sin (n * x) * n) x := by
      simpa using ((hasDerivAt_id' x).const_mul (n : ℝ)).cos
    have e1 := ((((hasDerivAt_id x).const_sub (π / 2)).mul hs).div_const (n : ℝ))
    have e2 := hc.div_const ((n : ℝ) ^ 2)
    convert e1.sub e2 using 1
    all_goals fa_close1b
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hF (by apply Continuous.intervalIntegrable; fun_prop)]
  simp only [mul_zero, Real.sin_zero, Real.cos_zero, Real.sin_nat_mul_pi]
  have hπ : π ≠ 0 := Real.pi_ne_zero
  field_simp
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.SeriesComputations.triangle_an
-- test: fa-triangle-an-test.sage

/-- The odd reciprocal square `1/(2m+1)²`. -/
noncomputable def oddInvSq (m : ℕ) : ℝ := 1 / (2 * (m : ℝ) + 1) ^ 2

/-- The tail of `Σ 1/(2k+1)²` from `k = M` is at most `1/(4M)`: the maximal error of `S_N` for the triangle wave
decreases like `1/N`. -/
theorem odd_tail_bound (M K : ℕ) (hM : 1 ≤ M) :
    ∑ k ∈ Finset.range K, oddInvSq (M + k) ≤ 1 / (4 * (M : ℝ)) := by
  have hM' : (1 : ℝ) ≤ M := by exact_mod_cast hM
  have key : ∀ K : ℕ, ∑ k ∈ Finset.range K, oddInvSq (M + k) ≤
      1 / (4 * (M : ℝ)) - 1 / (4 * ((M : ℝ) + K)) := by
    intro K
    induction K with
    | zero => simp
    | succ K ih =>
      rw [Finset.sum_range_succ]
      have hK : (0 : ℝ) ≤ K := Nat.cast_nonneg K
      have h4 : (0 : ℝ) < 4 * ((M : ℝ) + K) := by linarith
      have h5 : (0 : ℝ) < 4 * ((M : ℝ) + (K + 1)) := by linarith
      have hstep : oddInvSq (M + K) ≤ 1 / (4 * ((M : ℝ) + K)) - 1 / (4 * ((M : ℝ) + (K + 1 : ℕ))) := by
        simp only [oddInvSq]
        push_cast
        rw [div_sub_div _ _ h4.ne' h5.ne', div_le_div_iff₀ (by positivity) (mul_pos h4 h5)]
        nlinarith
      linarith
  have h := key K
  have : (0 : ℝ) ≤ 1 / (4 * ((M : ℝ) + K)) := by positivity
  linarith

example : veriphy_instance% KaluzaKlein.FourierAnalysis.SeriesComputations.odd_tail_bound (1) (0) (le_refl 1) := by
  norm_num [Finset.sum_range_succ, KaluzaKlein.FourierAnalysis.SeriesComputations.oddInvSq]

example : veriphy_instance% KaluzaKlein.FourierAnalysis.SeriesComputations.odd_tail_bound (1) (3) (le_refl 1) := by
  norm_num [Finset.sum_range_succ, KaluzaKlein.FourierAnalysis.SeriesComputations.oddInvSq]

example : veriphy_instance% KaluzaKlein.FourierAnalysis.SeriesComputations.odd_tail_bound (2) (4) ((by norm_num)) := by
  norm_num [Finset.sum_range_succ, KaluzaKlein.FourierAnalysis.SeriesComputations.oddInvSq]

example : veriphy_instance% KaluzaKlein.FourierAnalysis.SeriesComputations.odd_tail_bound (5) (2) ((by norm_num)) := by
  norm_num [Finset.sum_range_succ, KaluzaKlein.FourierAnalysis.SeriesComputations.oddInvSq]

attribute [veriphy_tested kernel_test 4 0] KaluzaKlein.FourierAnalysis.SeriesComputations.odd_tail_bound
-- test: fa-odd-tail-bound-test.sage

/-- The `L²` error of the square wave decays like `8/(πN)`, as stated (tested, not proved): with
`E(N) = (16/π) Σ_{odd n > N} 1/n²` (Parseval's tail), `|N E(N) - 8/π| ≤ 8/(πN)` for `N = 2M`, `M ≥ 1`. -/
def SquareL2Asymptotic : Prop :=
  ∀ M : ℕ, 1 ≤ M → |2 * M * (16 / π * ∑' k : ℕ, 1 / (2 * (M + k) + 1 : ℝ) ^ 2) - 8 / π| ≤ 8 / (π * M)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.SeriesComputations.SquareL2Asymptotic
-- test: fa-square-l2-asymptotic-test.sage

/-! ## exr-fs-half-range -/

/-- The odd extension of a function on `[0, π]`. -/
noncomputable def oddExt (f : ℝ → ℝ) (x : ℝ) : ℝ := if x ≤ 0 then -f (-x) else f x

/-- The odd extension's sine coefficient is the half-range one: `(1/π) ∫_{-π}^{π} f_o sin n x = (2/π) ∫_0^π f sin n x`. -/
theorem odd_extension_bn (f : ℝ → ℝ) (hf : Continuous f) (n : ℝ) :
    1 / π * ∫ x in (-π)..π, oddExt f x * Real.sin (n * x) = 2 / π * ∫ x in (0 : ℝ)..π, f x * Real.sin (n * x) := by
  have hl : ∫ x in (-π)..0, oddExt f x * Real.sin (n * x) = ∫ x in (-π)..0, -f (-x) * Real.sin (n * x) := by
    apply intervalIntegral.integral_congr_ae
    filter_upwards with x hx
    rw [Set.uIoc_of_le (by linarith [Real.pi_pos])] at hx
    simp [oddExt, hx.2]
  have hr : ∫ x in (0 : ℝ)..π, oddExt f x * Real.sin (n * x) = ∫ x in (0 : ℝ)..π, f x * Real.sin (n * x) := by
    apply intervalIntegral.integral_congr_ae
    filter_upwards with x hx
    rw [Set.uIoc_of_le (by linarith [Real.pi_pos])] at hx
    simp [oddExt, not_le.2 hx.1]
  have hsub : ∫ x in (-π)..0, -f (-x) * Real.sin (n * x) = ∫ x in (0 : ℝ)..π, f x * Real.sin (n * x) := by
    have := intervalIntegral.integral_comp_neg (a := -π) (b := 0) (fun x => f x * Real.sin (n * x))
    simp only [neg_neg, neg_zero, mul_neg, Real.sin_neg] at this
    rw [← this]
    apply intervalIntegral.integral_congr
    intro x _
    simp only [neg_mul]
  have hint1 : IntervalIntegrable (fun x => oddExt f x * Real.sin (n * x)) MeasureTheory.volume (-π) 0 := by
    apply (intervalIntegrable_congr_ae _).1 (show IntervalIntegrable (fun x => -f (-x) * Real.sin (n * x))
      MeasureTheory.volume (-π) 0 by apply Continuous.intervalIntegrable; fun_prop)
    rw [Filter.EventuallyEq, MeasureTheory.ae_restrict_iff' measurableSet_uIoc]
    filter_upwards with x hx
    rw [Set.uIoc_of_le (by linarith [Real.pi_pos])] at hx
    simp [oddExt, hx.2]
  have hint2 : IntervalIntegrable (fun x => oddExt f x * Real.sin (n * x)) MeasureTheory.volume 0 π := by
    apply (intervalIntegrable_congr_ae _).1 (show IntervalIntegrable (fun x => f x * Real.sin (n * x))
      MeasureTheory.volume 0 π by apply Continuous.intervalIntegrable; fun_prop)
    rw [Filter.EventuallyEq, MeasureTheory.ae_restrict_iff' measurableSet_uIoc]
    filter_upwards with x hx
    rw [Set.uIoc_of_le (by linarith [Real.pi_pos])] at hx
    simp [oddExt, not_le.2 hx.1]
  rw [← intervalIntegral.integral_add_adjacent_intervals hint1 hint2, hl, hr, hsub]
  ring

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.SeriesComputations.odd_extension_bn
-- test: fa-odd-extension-bn-test.sage

/-- The sines are the Dirichlet modes of `-d²/dx²` on `[0, π]`: `-(sin n x)'' = n² sin n x`, vanishing at `0` and `π`. -/
theorem sine_dirichlet_mode (n : ℕ) (x : ℝ) :
    deriv (deriv fun y => Real.sin (n * y)) x = -(n : ℝ) ^ 2 * Real.sin (n * x) ∧
      Real.sin (n * 0) = 0 ∧ Real.sin (n * π) = 0 := by
  have h1 : deriv (fun y => Real.sin (n * y)) = fun y => n * Real.cos (n * y) := by
    funext y
    exact (((hasDerivAt_id' y).const_mul (n : ℝ)).sin).deriv.trans (by ring)
  refine ⟨?_, by simp, Real.sin_nat_mul_pi n⟩
  rw [h1]
  exact ((((hasDerivAt_id' x).const_mul (n : ℝ)).cos).const_mul (n : ℝ)).deriv.trans (by ring)

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.SeriesComputations.sine_dirichlet_mode
-- test: fa-sine-dirichlet-mode-test.sage

/-- The cosines are the Neumann modes: `(cos n x)' = 0` at `0` and `π`. -/
theorem cosine_neumann_mode (n : ℕ) :
    deriv (fun y => Real.cos (n * y)) 0 = 0 ∧ deriv (fun y => Real.cos (n * y)) π = 0 := by
  have h1 : ∀ x, deriv (fun y => Real.cos (n * y)) x = -(n * Real.sin (n * x)) := by
    intro x
    exact (((hasDerivAt_id' x).const_mul (n : ℝ)).cos).deriv.trans (by ring)
  rw [h1, h1, Real.sin_nat_mul_pi]; simp

attribute [veriphy_tested oracle_test 0 3] KaluzaKlein.FourierAnalysis.SeriesComputations.cosine_neumann_mode
-- test: fa-cosine-neumann-mode-test.sage

/-- A function vanishing at both ends: the sine coefficient `∫_0^π x (π - x) sin x dx = 4`. -/
theorem parabola_b1 : ∫ x in (0 : ℝ)..π, x * (π - x) * Real.sin x = 4 := by
  veriphy_certified
    ( have hF :
        ∀ x_veriphy ∈ Set.uIcc (0 : ℝ) (π : ℝ),
          HasDerivAt
            (fun x_veriphy =>
              (-Real.pi * (x_veriphy * (Real.cos x_veriphy) - (Real.sin x_veriphy)) +
                    ((x_veriphy ^ 2 - 2) * (Real.cos x_veriphy)) -
                  (2 * x_veriphy * (Real.sin x_veriphy)) :
                ℝ))
            ((fun x => x * (π - x) * sin x) x_veriphy) x_veriphy :=
        by
        intro x_veriphy _
        beta_reduce
        refine
          ((((((((hasDerivAt_id' (x_veriphy : ℝ)).mul ((hasDerivAt_id' (x_veriphy : ℝ)).cos)).add
                                (((hasDerivAt_id' (x_veriphy : ℝ)).sin).const_mul (-1 : ℝ))).const_mul
                            (-Real.pi : ℝ)).add
                        ((((hasDerivAt_id' (x_veriphy : ℝ)).pow 2).add (hasDerivAt_const (x_veriphy : ℝ) (-2 : ℝ))).mul
                          ((hasDerivAt_id' (x_veriphy : ℝ)).cos))).add
                    (((hasDerivAt_id' (x_veriphy : ℝ)).mul ((hasDerivAt_id' (x_veriphy : ℝ)).sin)).const_mul
                      (-2 : ℝ)))).congr_of_eventuallyEq
                (Filter.Eventually.of_forall fun y => ?_)).congr_deriv
            ?_
        · beta_reduce; (simp; ring1)
        · (simp; ring1)
      rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (f := fun x_veriphy =>
          (-Real.pi * (x_veriphy * (Real.cos x_veriphy) - (Real.sin x_veriphy)) +
                ((x_veriphy ^ 2 - 2) * (Real.cos x_veriphy)) -
              (2 * x_veriphy * (Real.sin x_veriphy)) :
            ℝ))
          (f' := fun x => x * (π - x) * sin x) hF (by apply Continuous.intervalIntegrable; fun_prop)]
      beta_reduce
      have he0 : Real.cos (((0) : ℝ)) = 1 := by simp
      have he1 : Real.cos (((π) : ℝ)) = -1 := by
        rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring, Real.cos_nat_mul_pi]; norm_num
      have he2 : Real.sin (((0) : ℝ)) = 0 := by simp
      have he3 : Real.sin (((π) : ℝ)) = 0 := by rw [show (((π) : ℝ) : ℝ) = ((1 : ℕ) : ℝ) * Real.pi by push_cast; ring]; exact Real.sin_nat_mul_pi 1
      simp only [he0, he1, he2, he3]
      all_goals ring1)
  -- certificate: fa-integral.sage

end KaluzaKlein.FourierAnalysis.SeriesComputations
