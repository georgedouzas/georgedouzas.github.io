# Veriphy 0.3.0: gaps and defects found while certifying the book

Collected by the chapter agents (task 11.6); input for Veriphy's next release. Chapter reports in the commit messages list the claims that needed tests.

# Veriphy 0.3.0 pitfalls found while certifying (coordinator's notes, read before your next certify run)

1. A goal reached right after `have`/`generalize`/`refine ?_` is NOT translated (goal_kind 'other', goal.poly None:
   "the goal must be a polynomial equality"). Fix: put `rw [← sub_eq_zero]` (or `show <the goal>`) immediately
   before the `VERIPHY[...]` marker. A goal produced by `rw`, `refine ⟨?_, ?_⟩` bullets or `show` is fine.
2. Any `/` or `⁻¹` in ℂ (in a goal given to sage_run, or in the statement of a claim given to sage_test) together
   with a hypothesis in context makes Veriphy crash with "failed to synthesize LT ℂ" (it builds `0 < d` for the
   denominator). Fix: state claims without complex division (multiply out, use real angles
   `Complex.exp (((2 * π * k / n : ℝ) : ℂ) * I)`, put divisions in ℝ), and `generalize d⁻¹ = v` before markers.
3. Transcendental atoms (casts `((x : ℝ) : ℂ)`, `√3`, `Real.cos θ`, `Real.exp y`) become Sage functions, not
   polynomial generators, so `ideal_membership` sees no polynomial: `generalize ((√3 : ℝ) : ℂ) = s at h3 ⊢`,
   `generalize Real.cos θ = c at hc ⊢` etc. first (then rule 1: `rw [← sub_eq_zero]`).
4. Oracle instances: Sage's exact algebraic reals (AA, e.g. `abs(QQbar(3+4*I))`) are refused as "floating-point";
   wrap values in `RealIntervalField(128)(...)` (or compare with `ComplexIntervalField(128)` differences). For
   Gamma at complex points use `ComplexBallField(128)`.
5. A kernel instance whose proof derives a contradiction must say `exfalso` first when the goal is not `False`
   (linarith/omega do not close an arbitrary goal).
6. `scripts/lean_lock.py` crashed waiters with FileNotFoundError (race on slot removal); it is patched now. If one
   of your runs died with that traceback, simply rerun it.
7. (Fourier agent) `sage_test` dies with "unknown free variable `_fvar.N`" when the claim's statement divides by a
   bound variable inside a sum (`∑ k ∈ Finset.range K, 1 / (2 * (M + k) + 1) ^ 2`, `(-1)^i / (2 * i + 1)`): the domain
   condition for the denominator escapes the binder. Fix: wrap the summand in a definition (`oddInvSq m := 1/(2m+1)^2`)
   and sum `oddInvSq (M + k)`; the translator does not unfold definitions, and kernel instances close with
   `norm_num [Finset.sum_range_succ, <Namespace>.oddInvSq]`.
8. `ring` inside a `first | …` macro never fails: `ring` = `first | ring1 | ring_nf`, so a branch like `(simp …; ring)`
   "succeeds" leaving a normalised goal and the next branches are never tried. Use `ring1` in macros/alternatives.
9. Replacements printed by `sage_run` are re-flowed at 120 columns; `… by push_cast; ring]; exact Real.sin_nat_mul_pi 1`
   becomes `…];⏎  exact …`, which does not re-parse ("unexpected identifier; expected ')'"). Keep `rw […]` and the
   closing tactic on separate lines in the script (fa-trig-integral.sage does this now) and grep committed files for
   lines ending in `];`.
10. Never edit a Lean file while `certify.py` runs on it: certify rewrites the whole file at the end from the text it
    read at the start, so concurrent edits are lost.
11. `set_option … in` must come BEFORE the docstring (`set_option maxHeartbeats N in` / `/-- … -/` / `theorem …`);
    after the docstring it fails with "unexpected token 'set_option'; expected 'lemma'".
12. `matrix_from_goal` refuses matrices whose entries are trigonometric *values* (`cos 0`, `sin π`, `cos (2π)`): the
    relations cannot reduce an opaque `cos 0`. `.veriphy/certificates/fa-matrix-values.sage` (Sage checks the entries
    exactly, Lean uses `Real.cos_zero, Real.sin_pi, …`) handles `d¹(0) = 1`, `d^{1/2}(2π) = -1`.
13. A trig-polynomial integrand of degree 18 (the `L²` error of `S₉`) overflows `maxRecDepth` with the expand_trig
    antiderivative (62 monomials); `fa-trig-integral-ma.sage` keeps the antiderivative in multiple angles (16 terms) and
    needs `set_option maxHeartbeats 4000000 in` + `set_option maxRecDepth 4000 in`.
14. With `1/π` coefficients, `F' = f` may need `π * π⁻¹ = 1`: `linear_combination (norm := first | ring1 | (field_simp; try ring1))`
    with `have hπ : Real.pi ≠ 0` in context (fa-trig-integral.sage does this when π occurs).
15. The `derivative` recipe's "needs 't ^ 2 ≠ 0'" is fixed by `PROOFS = {'t ^ k ≠ 0': 'exact pow_ne_zero k ht'}` in a
    script copy; adding `have ht2 : t ^ 2 ≠ 0` hypotheses instead breaks the translation ("no derivative Veriphy
    translates").
16. The Sage daemon's per-script limit is 120 s (DEFAULT_TIMEOUT_S, not VERIPHY_SAGE_TIMEOUT): symbolic `integrate`
    over ℝ of `exp(-x²) e^{-ikx}` exceeds it; rigorous `ComplexBallField.integral` over [-10, 10] plus a tail bound takes
    seconds (fa-ft-deriv-test.sage).
