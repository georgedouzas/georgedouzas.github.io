# A Mathematical Journey to Relativity: Deriving Special and General Relativity with Basic Mathematics

Wladimir-Georges Boskoff and Salvatore Capozziello, Springer, 2024. Local copy: `references/textbooks/A Mathematical Journey to Relativity Deriving Special and General Relativity with Basic Mathematics.pdf` (git-ignored). Textbook (design D32).

## Chapters it serves

Mapped from the book's table of contents (chapter titles and page numbers of the PDF). The sections, equations
and problem numbers actually used are added below when the chapter that uses them is written or revised.

| Chapter of this book | Sections of the textbook | Use |
|---|---|---|
| part-i-09-differential-geometry | Ch. 4 (differential geometry of surfaces in 3D, p. 89ff), Ch. 5 (basic differential geometry, p. 144ff), Ch. 3 (Poincaré disk, p. 63ff) | content and examples |
| part-ii-08b-special-relativity | Ch. 9 (special relativity, p. 217-256) | content, examples and exercises |
| part-ii-09-general-relativity | Ch. 9-10 (special and general relativity, relativistic cosmology, p. 240-406) | content and exercises |
| part-vi-02-compactifications | Ch. 11-12 (de Sitter and anti-de Sitter space-times, p. 407-445): AdS backgrounds of Freund-Rubin and warped compactifications | content and examples |

## Sections, equations and problems used

Chapter 9, used in `part-ii-08b-special-relativity` (task 12.9). Boskoff and Capozziello use the signature $(+, -)$ in the
$(t, x^3)$ plane, as the chapter does.

- Sec. 9.1 (Michelson-Morley, postulates): def-sr-inertial-frame, exm-sr-frame-uniform, exr-sr-michelson-morley.
- Secs. 9.2-9.3 (Lorentz transformations in geometric coordinates and their consequences), cited for the Minkowski-spacetime
  section: def-sr-minkowski-spacetime.
- Theorem 9.1.7 (Lorentz transformation from the eigenvectors $(1, \pm c)$): thm-sr-boost (rewritten in light-cone coordinates).
- Exercise 9.1.8 (inverse transformation): exm-sr-boost-inverse. Theorem 9.1.9, Exercises 9.1.11 and 9.1.12 (composition, the
  velocity group, the abelian group of boosts): exr-sr-velocity-group, exr-sr-boost-group.
- Examples 9.1.4-9.1.6 (worldlines on a diagram): exr-sr-journey.
- Example 9.2.8 (twins, $v = 4/5$): exm-sr-twins, fig-sr-twin. Example 9.2.10 (contracted vehicle, $125\,$m at $3/5$): exm-sr-contraction-numbers.
- Theorem 9.2.14 (mass and energy): exm-sr-kinetic-expansion.
- Sec. 9.5 (Doppler effect), cited for the optics section: def-sr-wave-vector. Theorem 9.5.2 (Doppler effect, approaching
  and receding): prp-sr-doppler.
- Sec. 9.6 (constant gravitational field), cited for the dynamics section: def-sr-four-force. Proposition 9.6.1, Theorems 9.6.3 and 9.6.4 (uniformly accelerated frame, radar coordinates $t = e^{\alpha\bar x}\sinh(\alpha\tau)/\alpha$): prp-sr-hyperbolic-motion,
  exm-sr-radar-rindler. Exercise 9.6.5 (inverse map): exr-sr-radar-inverse (Sage). Theorem 9.6.6 (redshift $\nu e^{-\alpha h}$): exm-sr-boskoff-redshift,
  prp-sr-rindler-redshift.
- Not used: Sec. 9.4 (Maxwell equations), which the chapter takes from Blennow-Ohlsson Sec. 1.7 and Maggiore Ch. 8.

Chapters 3-5 and 7, used in `part-i-09-differential-geometry` (textbook pass, task 2). The book's surface notation
$f$, $N$, $h_{ij}$, $h^i_j$ and its Christoffel symbols $T^k_{ij}$ were translated to the chapter's $E$, $\mathbf n$, $h_{ij}$,
$S^i{}_j$, $\Gamma^k_{ij}$, and its Riemann symbols to the chapter's sign convention (both give $R_{1212} = \det h$).

| Section, example or problem | Block | Use |
|---|---|---|
| Ex. 4.3.3 (tractrix, pseudosphere metric) | `exm-hyp-pseudosphere`, `exr-hyp-tractrix` | embedding and induced metric $\cot^2\sigma\,d\sigma^2 + \sin^2\sigma\,d\varphi^2$ |
| Ex. 4.13.2 (pseudosphere to half-plane) | `def-hyp-half-plane` (intro), `exm-hyp-pseudosphere` | substitution $y = 1/\sin\sigma$ |
| Sec. 4.6, Exs. 4.6.1, 4.6.2 (second fundamental form, Weingarten formulas) | `def-sff-second-form`, `exm-sff-plane-sphere`, `def-sff-shape-operator` | |
| Problem 4.7.2 (cone) | `exr-sff-cone-torus` | |
| Sec. 4.8 (curvature as ratio of areas) | `exr-sff-gauss-map` | |
| Sec. 4.9, Thms. 4.10.1, 4.10.3 (Gauss formulas, Gauss equation, theorema egregium, Codazzi-Mainardi) | `thm-sff-gauss-equation`, `exr-sff-codazzi` | |
| Exs. 4.10.4, 4.10.5 (pseudosphere and sphere by the theorema egregium) | `exr-sff-pseudosphere`, `exm-sff-gauss-pseudosphere`, `exm-sff-gauss-sphere` | |
| Sec. 4.15 (central projection of the sphere onto a tangent plane) | `exr-gauss-gnomonic`, `exr-hyp-klein` (hyperbolic counterpart) | |
| Sec. 7.1 (Poincaré half-plane, its geodesics, non-secant lines, the disc) | `def-hyp-half-plane`, `prp-hyp-geodesics`, `exm-hyp-disc-diameter`, `exm-hyp-semicircle`, `exm-hyp-parallels`, `fig-hyperbolic-geodesics`, `prp-hyp-disc` | the disc is reached by the Cayley map instead of the book's inversion |
| Sec. 7.2 (stereographic projection of the Minkowski unit sphere onto the disc) | `exm-hyp-hyperboloid` | |

Not used in `part-i-09-differential-geometry` (task 8 check): Ch. 3 (geometric inversion, cross ratio, the Poincaré disc
built projectively), since the chapter reaches the disc by the Cayley map from the half-plane (Sec. 7.1 above) and the
projective construction adds no statement the chapter needs; Sec. 7.4 (the disc metric once more) for the same reason;
Ch. 5 (tensors, Riemann and Ricci tensors, covariant derivatives, the second Bianchi identity), which the chapter already
develops for general manifolds in its own conventions, so the book's coordinate treatment would repeat it; Secs. 4.11,
4.12 and 4.14 (Einstein's theorem for surfaces, parallel transport on surfaces, non-Euclidean ambient spaces), covered by
the chapter's general sections.

Ch. 10 (general relativity and relativistic cosmology), mapped to `part-ii-09-general-relativity`: not used in this pass.
The chapter's examples and exercises come from Blennow-Ohlsson Ch. 2, which covers the same ground (Schwarzschild orbits and
capture, black holes, de Sitter and anti-de Sitter, gravitational waves, FLRW cosmology) problem by problem; Boskoff and Capozziello's Ch. 10
was not in the pass's task list and its derivations would repeat the chapter's.

Chapters 11-12, used in `part-vi-01-consistent-reductions` and `part-vi-02-compactifications` (textbook pass, task 3). The
book's method (a quadric with normal $N = X/a$ is umbilic, $h_{ij} = g_{ij}/a$, and the Gauss equation with the sign of
$\langle N, N\rangle$ gives constant curvature and the cosmological constant) was translated to the book's curvature
conventions, in which anti-de Sitter space has $R_{\mu\nu} = +(D-1)L^{-2}g_{\mu\nu}$ in the signature $(+, -, \dots, -)$.
Caution: Sec. 12.5 writes anti-de Sitter space as $X_0^2 + X_1^2 - X_2^2 - \dots = -a^2$ in $M^{(2,n)}$; with the ambient
signature $(+, +, -, \dots)$ that quadric has a spacelike normal and induced signature $(2, n-3)$ (the book's own metric
$a^2dt^2 + a^2\sinh^2t\,dx_1^2 - a^2\cosh^2t\,dx_2^2$ shows it), so it is not a spacetime. The chapters use the quadric $X\cdot X = +L^2$
instead, and the book's "de Sitter slices of anti-de Sitter space" become the slices $X^D = \mathrm{const}$ of the correct quadric.

| Section, example or problem | Block | Use |
|---|---|---|
| Secs. 11.2, 12.5 (quadrics, umbilic second fundamental form, Gauss equation with the sign of $\langle N, N\rangle$, $R_{ij}$ and the cosmological constant $-(n-2)(n-3)/2a^2$ in $n-1$ dimensions) | `prp-vcp-ads-quadric`, `exm-vcp-adsq-three-quadrics`, `exr-vcp-adsq-lambda`, `exm-vcr-frg-quadric` | rewritten for $X \cdot X = +L^2$ in $\mathbb{R}^{2,D-1}$ |
| Secs. 11.2, 12.5 (recursive diagonal parametrisations of the quadrics) | `exr-vcr-frg-global` (Sage) | global coordinates of $AdS_4$ |
| Sec. 11.4 (flat slicing of de Sitter space) and Sec. 12.5 (slices of the anti-de Sitter quadric) | `exr-vcp-adsq-poincare` (Sage), `exr-vcp-adsq-ds-slicing` (Sage) | Poincaré coordinates; de Sitter slicing |
| Example 11.4.1 ($f(R) = R^2$ gravity and Einstein metrics, no cosmological term needed in four dimensions) | `exr-vcp-adsq-quadratic` | restated for $D$ dimensions |
