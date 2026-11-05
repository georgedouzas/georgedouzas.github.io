# Computational Quantum Mechanics

Joshua Izaac and Jingbo Wang, Springer, 2018. Local copy: `references/textbooks/Izaac and Wang - 2018 - Computational Quantum Mechanics.pdf` (git-ignored). Textbook (design D32).

## Chapters it serves

Mapped from the book's table of contents (chapter titles and page numbers of the PDF). The sections, equations
and problem numbers actually used are added below when the chapter that uses them is written or revised.

| Chapter of this book | Sections of the textbook | Use |
|---|---|---|
| part-ii-03-quantum-mechanics | Part III (solving the Schrödinger equation: one dimension, higher dimensions, time propagation, central potentials, p. 372ff) | Sage exercises |
| part-i-06-fourier-analysis | Ch. 8 (the Fourier transform, p. 324ff) | Sage exercises |
| part-i-07-hilbert-spaces-operators | Ch. 7 (the eigenvalue problem, p. 280ff) | exercises |

## Sections, equations and problems used

Chapter part-i-07-hilbert-spaces-operators (Functional analysis):

- Sec. 7.2 (power iteration), eqs. (7.19)-(7.21) (convergence of the Rayleigh quotient for a diagonal matrix), Sec. 7.2.1
  (inverse power iteration): @exm-fan-izaac-matrix, @exr-fan-power-iteration-rate, @exr-fan-izaac-hermitian.
- Problem P7.1 (the Hermitian $3 \times 3$ matrix): @exr-fan-izaac-hermitian.
- Problem P7.6 (the matrix with eigenvalues $3, -3, 2$): @exm-fan-izaac-matrix, @exr-fan-spectral-radius-sage, figure
  `fa2-spectral-radius`.

Chapter part-i-06-fourier-analysis (textbook pass, task 5). The book's symmetric $1/\sqrt{2\pi}$ convention is converted to
ours, $\hat f(k) = \int f e^{-ikx}dx$. PDF page = printed page + 15.

- Secs. 8.3.1-8.3.3, eqs. (8.31)-(8.35) (the DFT, orthogonality, one-sidedness, real data): def-fad-dft, exm-fad-four-points,
  exm-fad-cos-cubed.
- Problem P8.2 (orthogonality of the discrete exponentials): exr-fad-orthogonality.
- Sec. 8.5.1, eqs. (8.45)-(8.49) (Cooley-Tukey, butterfly): exm-fad-cooley-tukey; Problem P8.5 ($e^{\cos x}$, FFT against DFT):
  exr-fad-fft (Sage).
- Secs. 8.1-8.1.2, eqs. (8.7)-(8.19), Theorem 8.1 (discretised transform, Nyquist condition): prp-fad-sampling, exm-fad-alias-waves.
- Sec. 8.4.1 (aliasing): prp-fad-sampling, exm-fad-alias-waves; Sec. 8.4.2, eq. (8.43), Figs. 8.4-8.6 (leakage, Hann window):
  exm-fad-leakage.
- Sec. 8.3.4, eqs. (8.40)-(8.41) (centred DFT), Example 8.2 and Problem P8.1 (transform of the first excited oscillator state):
  exr-fad-oscillator-transform (Sage).
- Sec. 8.2, eqs. (8.20)-(8.25), Problems P8.3-P8.4 (Fourier differentiation of $\psi_1$ and $\psi_6$): exr-fad-spectral-derivative (Sage).
- Errors noted: in eqs. (8.9), (8.19) and (8.37) the prefactors $\Delta x$ and $\Delta k$ are interchanged (the transform
  $\phi_m$ carries $\Delta x/\sqrt{2\pi}$), and Example 8.2 multiplies by `dk`; Theorem 8.2 and eq. (8.36) should read $N \geq 2m_c$,
  not $N = 1/2m_c$; Problem P8.1(a) uses the prefactor $1/2\pi$ instead of the $1/\sqrt{2\pi}$ of eq. (8.4).

Chapter part-ii-03-quantum-mechanics (textbook pass, task 5).

- Secs. 9.5-9.6, eqs. (9.54)-(9.67) (Numerov, the finite-difference matrix): prp-qmn-grid-box, exm-qmn-small-grids, exm-qmn-grid-error,
  exm-qmn-numerov, exr-qmn-grid-positivity.
- Problem P9.7 (direct matrix method, error against $\Delta x$): exr-qmn-finite-well-matrix (Sage), exr-qmn-morse (Sage).
- Problems P9.4 (Morse potential, $E_0 = (251 + 4\sqrt{15})/40$) and P9.8 (Numerov matrix): exr-qmn-morse (Sage); P9.8 also exm-qmn-numerov.
- Secs. 11.2-11.2.2, eqs. (11.9)-(11.17) (Euler, backward Euler, Crank-Nicolson): prp-qmn-time-step, exm-qmn-euler-spin, exm-qmn-cn-grid,
  exr-qmn-backward-euler.
- Sec. 11.3, eqs. (11.18)-(11.20) (split operator): prp-qmn-time-step, exm-qmn-split-force; Problem P11.1 (local error of the
  symmetric splitting): exr-qmn-strang.
- Problems P11.2-P11.3 (free Gaussian packet, Euler against a second-order scheme): exr-qmn-cn-packet (Sage).
- Sec. 10.3, Theorem 10.1, eqs. (10.32)-(10.38) (variational principle): prp-qmv-variational, exm-qmv-box-parabola, exm-qmv-oscillator,
  exm-qmv-hydrogen-gaussian, exr-qmv-quartic (Sage).
- Problem P10.9 (non-orthogonal basis, $Hc = ESc$): exr-qmv-generalised. Sec. 10.2.1 and Problem P10.8 (Pöschl-Teller well,
  oscillator basis; here in one dimension): exr-qmv-poschl-teller (Sage).
- Errors noted: in Sec. 10.3, after eq. (10.35), "$E_n - E_0 \leq 0$ for all $n \leq 0$" should read "$\geq 0$ for all $n \geq 0$";
  Problem P9.9 should say $|x| \geq L/2$, not $|x| \geq L$.
- Not used: Ch. 12 (central potentials, numerical radial equation) and Ch. 13 (Hartree-Fock), which repeat the hydrogen
  results of the chapter numerically; Sec. 11.5 (Chebyshev propagation). Also not used (task 8 check): Secs. 9.1-9.4
  (boundary value problems and plain shooting, superseded in the chapter by Numerov and the matrix method), Sec. 10.1
  (two-dimensional matrix method, a direct extension of prp-qmn-grid-box), Secs. 11.1, 11.4 and 11.6 (the unitary
  propagator, already in the chapter; direct time discretisation and the Nyquist condition, which add no statement).
- The intro to the chapter's part on Banach algebras cites Ch. 7 for its numerical examples (before def-banach-algebra);
  Problem P7.6 is also cited in the caption of the figure `fa2-spectral-radius` and before def-algebra-spectrum; Secs.
  7.1, 7.3-7.6 (eigenvalue basics, Krylov methods, conditioning, library interfaces) and Secs. 8.6-8.7 (library
  interfaces) are not used, being numerical-library material without statements for the chapters.
