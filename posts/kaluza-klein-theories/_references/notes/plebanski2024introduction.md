# An Introduction to General Relativity and Cosmology

Jerzy Plebański and Andrzej Krasiński, Cambridge University Press, 2024. Local copy: `references/textbooks/Plebanski and Krasinski - 2024 - An Introduction to General Relativity and Cosmolog.pdf` (git-ignored). Textbook (design D32).

## Chapters it serves

Mapped from the book's table of contents (chapter titles and page numbers of the PDF). The sections, equations
and problem numbers actually used are added below when the chapter that uses them is written or revised.

| Chapter of this book | Sections of the textbook | Use |
|---|---|---|
| part-ii-09-general-relativity | Part I (general relativity: tensors, curvature, Einstein equations, Schwarzschild) and Part II (cosmology: Friedmann and inhomogeneous models) | content and exercises |
| part-vi-01-consistent-reductions | the chapter on spatially homogeneous Bianchi-type spacetimes (a separate file in the PDF's outline): group-manifold metrics | examples |

## Sections, equations and problems used

Exercises adapted in `part-ii-09-general-relativity` (task 12.8), cited as `[@plebanski2024introduction, Sec. N.17, Exercise n]`:

| Exercise | Block | Use |
|---|---|---|
| Sec. 12.19, Exercises 1-2 | `exr-grb-newtonian-fluid` | Newtonian limit of the conservation law: Euler and continuity equations |
| Sec. 14.17, Exercise 7 | `exr-grb-frame-curvature` | orthonormal-frame Riemann components of Schwarzschild, regular at r = 2m (Sage) |
| Sec. 17.17, Exercise 1 | `exr-grb-rw-einstein` | Einstein tensor of the Robertson-Walker metric is diagonal (Sage) |
| Sec. 17.17, Exercise 2 | `exr-grb-radial-null` | radial null geodesics stay radial |

Not used in `part-ii-09-general-relativity` (task 8 check): the text of Parts I-II, which the chapter develops in its own
order and conventions (the book serves for exercises), and the inhomogeneous cosmological models of Part II
(Lemaître-Tolman, Szekeres), beyond the chapter's Robertson-Walker statements.

Chapter 10 (spatially homogeneous Bianchi-type spacetimes, the separate PDF chapter), used in
`part-vi-01-consistent-reductions` (textbook pass, task 3). The structure constants $C^l{}_{ij}$ and Killing fields $k_{(i)}$
of the book become the chapter's $f_{ab}{}^c$ and right-invariant fields $\xi_a$; the invariant fields $X_{(i)}$ become the
left-invariant frame $Z_a$.

| Section, equation or exercise | Block | Use |
|---|---|---|
| Sec. 10.1, eqs. 10.5-10.18, Tables 10.1-10.2 (normal form, Jacobi condition $n^{ij}a_j = 0$, rescalings, the types) | `prp-vcr-bianchi-classification`, `exm-vcr-bc-met`, `exm-vcr-bc-sl2`, `exr-vcr-bc-type-iv`, `exr-vcr-bc-jacobi` (Sage), `exr-vcr-bc-invariant`, `exr-vcr-bc-sol` | |
| Sec. 10.2 (three-dimensional groups with two-dimensional orbits: IX, VII$_0$, VIII) | `exm-vcr-bc-orbits` | |
| Secs. 10.4-10.6, eq. 10.31 (simply transitive action, the Bianchi metrics $dt^2 - g_{(j)(k)}(t)\omega^{(j)}\omega^{(k)}$) | `exm-vcr-gm-bianchi-cosmology` | as a group-manifold reduction to one dimension |
| Sec. 10.7 (Bianchi types inside Robertson-Walker: IX; I, VII$_0$; V, VII$_h$) | `exm-vcr-gm-bianchi-cosmology` | |
| Sec. 10.5, eq. 10.26 and Sec. 10.8, Exercise 3 ($D^l{}_{ij} = -C^l{}_{ij}$) | `exr-vcr-gm-right-invariant` (Sage) | Heisenberg group |
| Sec. 10.8, Exercise 4 (time-dependent invariant frames) | `exr-vcr-gm-time-frames` | |
