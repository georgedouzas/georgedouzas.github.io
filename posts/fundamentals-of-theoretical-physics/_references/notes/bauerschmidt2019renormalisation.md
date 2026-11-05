# Introduction to a Renormalisation Group Method

Roland Bauerschmidt, David C. Brydges and Gordon Slade, Springer (Lecture Notes in Mathematics 2242), 2019. Local copy: `references/textbooks/Bauerschmidt et al. - 2019 - Introduction to a Renormalisation Group Method.pdf` (git-ignored). Textbook (design D32).

## Chapters it serves

Mapped from the book's table of contents (chapter titles and page numbers of the PDF). The sections, equations
and problem numbers actually used are added below when the chapter that uses them is written or revised.

| Chapter of this book | Sections of the textbook | Use |
|---|---|---|
| renormalisation (new Part II chapter, D31) | Part I (spin systems and critical phenomena, p. 13ff), Part II (perturbative renormalisation group, p. 77ff), Part III outline (nonperturbative analysis, p. 114ff) | content for the Wilsonian and rigorous renormalisation group |
| part-ii-02-thermodynamics-statistical-mechanics | Part I (spin systems, Gaussian free field, critical exponents) | examples and exercises |

## Sections, equations and problems used

### part-ii-08-renormalisation

- Sec. 1.1 (critical phenomena and the renormalisation group): sec-wilsonian-rg introduction.
- Eq. 1.5.25, Sec. 1.5.3, Exercises 1.5.4-1.5.5 (lattice Laplacian, bubble diagram, recurrence): def-ren-lattice-regularisation,
  exm-ren-lattice-bubble, exr-ren-lattice-bubble, exr-ren-lattice-c00.
- Sec. 1.6.2, eqs. 1.6.10-1.6.11, Thm. 1.6.1, eq. 1.7.2 (critical behaviour above, at and below four dimensions, logarithmic
  corrections, self-avoiding walk): exm-ren-fp-gaussian, exm-ren-wf-four, exm-ren-pi-theorem, exm-ren-ce-saw (Sec. 1.7 cited in its introduction).
- Prop. 2.1.6, Def. 2.1.8, Cor. 2.1.11, eq. 2.1.21, Sec. 5.1 (Gaussian convolution, Wick ordering, progressive integration,
  reduction to block analysis):
  def-ren-progressive-integration, exm-ren-pi-convolution, exm-ren-pi-wick, exr-ren-pi-wick.
- Eqs. 6.1.1-6.1.3, Prop. 6.1.3, Exercise 6.1.1 (perturbative flow of the coupling): exm-ren-pi-flow, fig-ren-flow-g, exr-ren-pi-recursion;
  Prop. 6.1.3 (logarithmic estimate of the flow): exr-ren-pi-log. The hierarchical model of Ch. 4 enters only as the
  setting of eq. 6.1.1 in the prose and in fig-ren-flow-g.
- Not used (task 8 check): Ch. 3 (finite-range decomposition), the rest of Ch. 4 (hierarchical GFF and $|\varphi|^4$
  model), Secs. 5.2-5.3 and 6.2 (the renormalisation group map and the reduction of the main theorem), Part III
  (Chs. 7-10, nonperturbative analysis, of which the mapping took only the outline) and Part IV (self-avoiding walk and
  supersymmetry): rigorous constructions whose norms and estimates are beyond the chapter, which states their outcome
  only.

### part-ii-02-thermodynamics-statistical-mechanics

Added in task 7 of change kk-csdr-textbook-pass, in Phase transitions and Landau theory (new units The Curie-Weiss
model, Correlation length, The Gaussian free field). Material already used by part-ii-08-renormalisation (the
bubble diagram, Exercises 1.5.4-1.5.5, Ch. 2 Gaussian integration, Sec. 1.6.2) is cross-referenced, not repeated.

- Sec. 1.2, eqs. 1.2.1-1.2.2 (Ising energy as a gradient energy): exr-gff-ising-gradient.
- Sec. 1.2, eqs. 1.2.5-1.2.10 (two-point function, correlation length, susceptibility, exponents $\nu$, $\eta$, Fisher's
  relation, two-dimensional values): def-correlation-length, exm-crl-susceptibility-sum, exm-crl-fisher, exr-crl-ising-chain.
- Sec. 1.3 and eqs. 1.6.3-1.6.4 (spin systems classified by the single-spin distribution, universality): exm-gff-single-spin.
- Sec. 1.4.1, eq. 1.4.4 (mean-field model, energy a function of the mean spin): prp-curie-weiss-potential,
  exm-cwm-entropy, exr-cwm-finite-sum.
- Lemmas 1.4.2, 1.4.3 and 1.4.6 (renormalised measure and potential, fluctuation measure, convexity): prp-curie-weiss-potential,
  exm-cwm-renormalised-potential, exm-cwm-order-of-limits, exr-cwm-convexity.
- Thm. 1.4.1, Thm. 1.4.4, eq. 1.4.23 (exact mean-field exponents, Laplace's principle): prp-curie-weiss-exact,
  exm-cwm-magnetisation, exr-cwm-finite-size, exr-cwm-critical-window; Exercise 1.4.5: exr-cwm-laplace.
- Exercises 1.4.7 and 1.4.8 ($n$-component mean-field model, $\beta_c = n$): exr-cwm-heisenberg.
- Eq. 1.6.8 (correlation length of order $p$): exr-crl-second-moment.
- Def. 1.5.1, Exercise 1.5.2, eqs. 1.5.3-1.5.4 (Gaussian free field, positivity, zero mode): prp-gff-covariance,
  exm-gff-positivity, exr-gff-ring.
- Lemma 1.5.3, eqs. 1.5.11-1.5.21 (random-walk representation, GFF exponents): prp-gff-random-walk, exm-gff-killed-walk,
  exr-gff-first-step, exr-gff-walk-sum, exr-gff-green-decay; Exercise 1.5.5 (recurrence): exm-gff-polya; Exercise 1.5.6
  (intersections of two walks and the bubble): exr-gff-intersections.
- Not used here: Ch. 2.2 (cumulants; the probability chapter covers them), Ch. 3 (finite-range decomposition) and Ch. 4
  (hierarchical model), which belong to the renormalisation group and are outlined in part-ii-08-renormalisation.

Errata found:
- Eqs. 1.3.1, 1.4.2 and 1.4.4 put $+h\cdot\varphi_x$ into the energy $H$, whose weight is $e^{-H}$, while eq. 1.2.4, eqs.
  1.4.13 and 1.4.17 and Thm. 1.4.1 ($M(\beta_c, h) \sim (3h)^{1/3}$ for $h \downarrow 0$) need the field to favour
  alignment: the energy should contain $-h\cdot\varphi_x$. The chapter uses $H = \dots - h\sum_is_i$.
- Exercise 1.5.4 gives $b_1 = \frac18$; in one dimension $C_{0x} \approx e^{-m|x|}/(2m)$, so $B_{m^2} \sim \frac{1}{4m^3}$ and
  $b_1 = \frac14$ ($b_2$, $b_3$, $b_4$ are correct).
