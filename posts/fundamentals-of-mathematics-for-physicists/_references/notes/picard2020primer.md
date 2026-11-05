# A Primer for a Secret Shortcut to PDEs of Mathematical Physics

Rainer Picard, Des McGhee, Sascha Trostorff and Marcus Waurick, Birkhäuser, 2020. Local copy: `references/textbooks/A Primer For A Secret Shortcut To PDEs Of Mathematical Physics.pdf` (git-ignored). Textbook (design D32).

## Chapters it serves

Mapped from the book's table of contents (chapter titles and page numbers of the PDF). The sections, equations
and problem numbers actually used are added below when the chapter that uses them is written or revised.

| Chapter of this book | Sections of the textbook | Use |
|---|---|---|
| part-i-07-hilbert-spaces-operators | Appendix B (requisites from functional analysis, p. 139ff): Hilbert spaces, closed and skew-selfadjoint operators; Ch. 1 (solution theory of evolutionary equations, p. 12ff): well-posedness through operator theory | content for the functional-analysis parts (D31) |
| part-i-05-differential-equations | Ch. 2 (applications: heat, wave, Maxwell and elasticity equations, p. 42ff) | examples |

## Sections, equations and problems used

Chapter part-i-07-hilbert-spaces-operators (Functional analysis):

- Def. 1.1.1 (the weighted space $H_{\rho,0}(\mathbb{R})$) and Thm. 1.1.4 ($\|\partial_{0,\rho}^{-1}\| \leq 1/\rho$):
  @exm-fan-time-derivative, @exr-fan-time-derivative-bound.
- Lemma B.1.5 (Cauchy-Schwarz from the Gram determinant; the norm of an inner product) and Lemma B.1.7 with the remark after
  it (parallelogram law and its converse): @exm-fan-inner-product-norm, @exr-fan-parallelogram.
- Def. B.4.1 (closed and closable operators), Lemma B.4.2 (graph inner product; closability criterion): @def-closed-operator,
  @exm-fan-not-closable, @exr-fan-graph-norm, @exr-fan-closable-criterion.
- Lemma B.4.3 ($B(H_0, H_1)$ is a Banach space): @def-bounded-linear-map.
- Sec. B.4 (linear operators and adjoints), cited with Def. B.4.1 in the introduction of the part on the four
  principles of functional analysis (before thm-hahn-banach); App. B as a whole is cited in the chapter's introduction.
- Not used (task 8 check): Secs. 1.2-1.3 (ODEs in the weighted space, Picard's theorem for evolutionary equations),
  which need the Fourier-Laplace transform and skew-selfadjoint operators beyond the chapter's statements; Secs. B.2-B.3
  (projection and Riesz theorems), which the chapter takes from its other sources in its own order; skew-selfadjoint
  operators (Sec. B.4 after Lemma B.4.3), not needed by any statement of the chapter.

Chapter part-i-05-differential-equations (textbook pass, task 6):

- Sec. 2.1 (the operator A = (0, div; grad, 0) and its skew-symmetry, Proposition 2.1.3): exm-wave-first-order.
- Sec. 2.1.1, eqs. (2.1.7)-(2.1.8) (heat equation as a first-order system with Fourier's law): exm-heat-flux-system.
- Sec. 2.1.2, eqs. (2.1.9)-(2.1.12) (Maxwell-Cattaneo-Vernotte law, damped wave equation): exr-class-telegraph.
- Not used: Secs. 2.2-2.4 (relativistic Schrödinger equation, linear elasticity, Guyer-Krumhansl model), whose
  operator-theoretic setting belongs to the functional-analysis chapter rather than to the elementary PDE sections;
  Secs. 2.5-2.6 (Maxwell's equations and coupled phenomena as evolutionary equations) and Ch. 3 for the same reason.
