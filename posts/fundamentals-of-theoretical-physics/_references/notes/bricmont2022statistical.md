# Making Sense of Statistical Mechanics

Jean Bricmont, Springer, 2022. Local copy: `references/textbooks/Bricmont - 2022 - Making Sense of Statistical Mechanics.pdf` (git-ignored). Textbook (design D32).

## Chapters it serves

Mapped from the book's table of contents (chapter titles and page numbers of the PDF). The sections, equations
and problem numbers actually used are added below when the chapter that uses them is written or revised.

| Chapter of this book | Sections of the textbook | Use |
|---|---|---|
| part-ii-02-thermodynamics-statistical-mechanics | Ch. 5 (thermodynamics, p. 114ff), Ch. 6 (equilibrium statistical mechanics, p. 153ff), Ch. 9 (phase transitions, p. 282ff), exercises with hints in Ch. 11 (p. 323ff) | content and exercises |
| part-i-08-probability | Ch. 2 (probability, p. 16ff) | examples and exercises |

## Sections, equations and problems used

### part-ii-08-renormalisation

- Sec. 9.9 (critical points, mean-field exponents, eq. 9.9.8 exact two-dimensional Ising exponents, eq. 9.9.9 Yang's
  magnetisation): sec-wilsonian-rg introduction, fig-ren-exponents, exm-ren-ce-two.
- Exercise 9.7 (the power 1/8 in eq. 9.9.9): exr-ren-ce-yang.

### part-ii-02-thermodynamics-statistical-mechanics (task 12.8)

Exercises of Chs. 5, 6 and 9 (hints in Ch. 11), cited as `[@bricmont2022statistical, Exercise n]`:

- Exercise 5.2 (heat in isothermal, isochoric and isobaric processes): exm-tsb-heat-three-processes.
- Exercise 5.3 (net heat of a three-step cycle): exr-tsb-cycle.
- Exercise 5.5 (thermodynamics from F = -CVT^4): exm-tsb-photon-free-energy.
- Exercise 5.6 (Maxwell relations of the grand potential): exm-tsb-grand-maxwell.
- Exercise 5.7 (exact and inexact differential forms): exm-tsb-exact-forms.
- Exercise 5.8 (Carnot cycle of one mole between 200 K and 100 K; only the volume ratio 4 is used, as in the hint): exm-tsb-carnot-mole.
- Exercise 5.9 (Legendre transforms of x^alpha and e^x): exr-tsb-legendre.
- Exercise 5.11 (equations of state from U(S, V, N), case 1): exr-tsb-equations-of-state.
- Exercise 5.12 (fundamental relation of E = aVT^4, P = E/3V): exr-tsb-fundamental (Sage).
- Exercises 6.3-6.5 (spins: s(e), f(T) and their Legendre duality): exr-tsb-spin-legendre (Sage).
- Exercises 6.6-6.8 (oscillators: s(e), f(T) and their Legendre duality): exm-tsb-einstein-legendre.
- Exercise 6.9 (defects in a crystal): exm-tsb-defects.
- Exercise 6.10 (para and ortho hydrogen, four-state toy model): exr-tsb-ortho-para.
- Exercise 6.11 (microcanonical classical oscillators): exr-tsb-oscillator-volume.
- Exercise 6.12 (one-body potentials, PV = NkT from the grand partition function): exr-tsb-one-body.
- Exercise 9.9 (spin-one Ising chain by transfer matrix): exm-tsb-spin-one-mean-field (mean-field comparison), exr-tsb-spin-one-chain (Sage).
- Exercise 9.10 (lattice-gas form of the Ising chain): exm-tsb-lattice-gas, exr-tsb-lattice-gas (Sage).
- Not used (task 8 check): the text of Chs. 5, 6 and 9, which the chapter covers in its own order from its other
  sources (the book is used for its exercises); Exercises 5.1, 5.4 and 5.10 (checking the book's own formulas and the
  entropy of the ideal gas, already examples of the chapter), 6.1-6.2 (estimates and ensemble equivalence in the
  book's notation), 9.1-9.6 (Gibbs states, cluster bounds, the Peierls argument in three dimensions, rigorous material
  beyond the chapter's statements) and 9.8 (mean-field free energy, already derived in the chapter); Exercise 9.7 is used
  in part-ii-08-renormalisation (above).

### part-i-08-probability (textbook pass, task 6)

- Sec. 2.2.5 (Bertrand's paradox): exm-pbr-bertrand.
- Secs. 2.3.1-2.3.2 (typical sets, microstates and macrostates, laws of large numbers): exm-pbr-typical.
- Exercise 2.1 (at least four heads in six tosses): exm-pbr-four-heads.
- Exercise 2.8 (the first person unluckier than you): exr-pbr-unlucky.
- Exercise 2.10 (frequencies of a biased coin, large deviations): exr-pbr-large-deviations (Sage).
- Exercise 2.11 (de Moivre-Laplace from Stirling's formula): exr-pbr-stirling.
- Exercise 2.12 (multinomial distribution): exr-pbr-multinomial.
- Exercise 2.13 (drunken sailor): exr-rw-sailor.
- Exercise 2.14.1 (Bertrand's box paradox): exr-pbr-bertrand-box; Exercise 2.15 (Bayesian updating with three coin
  types): exr-pbr-three-coins.
- Errors noted: after eq. (2.3.4) the binomial coefficient is said to peak at N0 = 1/2; it peaks at N0 = N/2.
- Not used: Secs. 2.2.1-2.2.4, 2.4-2.6 (interpretations of probability, discussed but without computations for the
  chapter), Exercises 2.2 and 2.14.2 (birthday and Monty Hall problems, already in the chapter), 2.3-2.7 and 2.9
  (Gaussian moments, Poisson limit, binomial moments and St Petersburg game, already in the chapter), 2.16-2.20
  (measure theory of Appendix 2.A).
