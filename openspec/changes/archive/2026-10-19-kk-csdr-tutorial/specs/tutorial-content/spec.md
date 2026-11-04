# Spec Delta

## Purpose

Defines the pedagogical structure and content obligations of the Kaluza-Klein and CSDR tutorial, so that a reader
with undergraduate physics and multivariable calculus can follow every derivation from first principles to the
four-dimensional theory obtained from a sphere as internal space.

## ADDED Requirements

### Requirement: Chapter order builds from geometry to CSDR to the survey
The tutorial SHALL be organised in six parts in this order: mathematics, physics, abelian Kaluza-Klein theory,
non-abelian Kaluza-Klein theory, coset space dimensional reduction, and the survey of compactification methods.
Each part SHALL use only concepts defined in earlier parts or in itself.

#### Scenario: Contents of the two foundation parts
- **WHEN** the table of contents is read
- **THEN** the mathematics part contains, in this order, sets and topology, linear algebra, real analysis, complex analysis, differential equations, Fourier analysis, Hilbert spaces and operator theory, probability and statistics, manifolds, vector fields and tensors, metrics, connections and curvature, flows and Killing vectors, groups, Lie groups and algebras, representations, coset spaces and bundles; and the physics part contains, in this order, classical mechanics, thermodynamics and statistical mechanics, quantum mechanics, fields, actions and symmetries, gauge theories, quantum field theory, symmetry breaking, general relativity, supersymmetry and supergravity

#### Scenario: Table of contents reflects the progression
- **WHEN** a reader opens the rendered tutorial's table of contents
- **THEN** the six parts appear in the stated order, and each later part's chapters cite the earlier definitions they rely on by cross-reference

#### Scenario: A concept is used before its definition
- **WHEN** a chapter uses a notion (for example a Killing vector or a structure constant) that is not defined in that chapter or an earlier one
- **THEN** the tutorial is non-conforming and the definition is added where the progression requires it

### Requirement: Prerequisites and conventions are stated once
The tutorial SHALL open with a prerequisites statement that assumes only undergraduate physics and multivariable
calculus, and a conventions section fixing signature, index ranges, the Riemann sign, generator and structure
constant normalisation, the coupling convention and the units. Every chapter and every Lean statement SHALL use
these conventions.

#### Scenario: Advanced topics are taught, not assumed
- **WHEN** the prerequisites statement is read
- **THEN** it does not assume field theory, general relativity, supersymmetry or supergravity, which the foundations part teaches

#### Scenario: Reader checks a sign convention
- **WHEN** a reader needs the sign convention behind a displayed curvature or field-strength expression
- **THEN** the conventions section states it and the displayed expression agrees with it

#### Scenario: Lean statement uses the same conventions
- **WHEN** a displayed result is compared with the Lean theorem that verifies it
- **THEN** the theorem's definitions encode the same signature, sign and normalisation choices as the conventions section

### Requirement: No derivation step is omitted
Every displayed equation that follows from earlier equations SHALL be reachable from the displayed intermediate
expressions by a single named operation (substitution, contraction, differentiation, integration by parts, use of a
stated identity). Phrases that skip work, such as "after a straightforward calculation", SHALL NOT replace steps.

#### Scenario: Reduction of the five-dimensional curvature scalar
- **WHEN** the tutorial derives the four-dimensional action from the Kaluza-Klein metric ansatz
- **THEN** the inverse metric, every Christoffel symbol, every Ricci tensor component and the curvature scalar are displayed before the action is written

#### Scenario: Solving the CSDR constraints on the sphere
- **WHEN** the tutorial solves the symmetric-field constraints for `SU(2)/U(1)`
- **THEN** the embedding of the isotropy generator, the decomposition of the gauge algebra under it, the resulting constraint on each field component and the surviving fields are each displayed

### Requirement: Every abstract definition has worked examples
Each definition in the mathematics and physics parts SHALL be followed by explicit computations, as many as the material
allows and at least three, on the running examples (`S^1`, `S^2`, `SU(2)`; the `SU(2)` Yang-Mills field; explicit
metrics for the Einstein equations; explicit gamma matrices for the superalgebra), carried through to numbers or
closed-form components.

#### Scenario: Christoffel symbols introduced
- **WHEN** the Levi-Civita connection is defined
- **THEN** all Christoffel symbols of the round metric on `S^2` are computed in spherical coordinates and displayed

#### Scenario: Killing vectors introduced
- **WHEN** Killing vectors are defined
- **THEN** the three Killing vector fields of `S^2` are given in coordinates, their Lie brackets are computed and shown to close into `su(2)`

#### Scenario: Yang-Mills field strength introduced
- **WHEN** the non-abelian field strength is defined
- **THEN** it is computed for an explicit `SU(2)` gauge field configuration, component by component, and the Bianchi identity is checked on it

#### Scenario: Einstein equations introduced
- **WHEN** the Einstein equations are derived from the Einstein-Hilbert action
- **THEN** they are checked on an explicit metric (a maximally symmetric space with cosmological constant), with every curvature component displayed

#### Scenario: Superalgebra introduced
- **WHEN** the four-dimensional N=1 supersymmetry algebra is defined
- **THEN** its anticommutators are verified on explicit gamma matrices and the Wess-Zumino model's supersymmetry variations are checked to leave the action invariant up to a total derivative

### Requirement: Group theory is taught extensively, with cosets as a central example
The mathematics part SHALL contain chapters on group theory (groups, subgroups, cosets and quotient groups,
homomorphisms, representations, Schur's lemma), on Lie groups and Lie algebras, and on representations of Lie
algebras (roots, weights, tensor products, branching rules), each with finite and matrix examples and exercises.

#### Scenario: Cosets introduced
- **WHEN** cosets and quotient groups are defined
- **THEN** they are worked on a finite group (for example `Z_6/Z_3` or `S_3/A_3`) and on `SO(2) ⊂ SO(3)`, and the reader is told how the sphere arises as the quotient

#### Scenario: Schur's lemma introduced
- **WHEN** Schur's lemma is stated
- **THEN** it is applied to an explicit representation (the commutant of an `su(2)` irreducible representation is computed and shown to be the scalars) and its role in the CSDR rules is announced

#### Scenario: Branching rules introduced
- **WHEN** branching rules are defined
- **THEN** the decomposition of the adjoint of `su(3)` under `su(2) × u(1)` and under `u(1) × u(1)` is computed from the weights and displayed

### Requirement: A dedicated coset-space chapter with many worked examples
The mathematics part SHALL contain a chapter on coset spaces with at least six worked examples, the reductive and
symmetric decompositions with the structure-constant split, the isotropy representation and its charges, coset
vielbeins and the metric from the Maurer-Cartan forms, and exercises solved in Sage or Lean.

#### Scenario: The examples
- **WHEN** the coset chapter's examples are listed
- **THEN** they include `S^n = SO(n+1)/SO(n)`, `S^3` as a group and as a symmetric coset of `SO(4)`, `CP^n`, the flag manifold `SU(3)/U(1)×U(1)`, the nearly-Kähler six-dimensional cosets and the torus

#### Scenario: Reader looks up a coset
- **WHEN** a reader looks for one of the listed cosets
- **THEN** its group, isotropy group, dimension, reductive decomposition, whether it is symmetric, and its isotropy representation are stated, and at least one of these facts is verified in Lean or Sage

#### Scenario: Coset vielbein constructed
- **WHEN** the coset vielbein is defined
- **THEN** it is computed for `S^2 = SU(2)/U(1)` from a coset representative and shown to reproduce the round metric, with the `U(1)` connection identified

### Requirement: One definition per block, each with at least three examples
Each definition block SHALL define exactly one notion (no stacked definitions such as "chart, atlas, manifold" in one
block), and SHALL be followed by at least three worked examples, with a figure where the notion can be drawn.

#### Scenario: A stacked definition
- **WHEN** a definition block names more than one defined term in its title or defines several notions in its body
- **THEN** the chapter is non-conforming and the block is split, each part with its own examples

#### Scenario: A definition with too few examples
- **WHEN** a definition block is followed by fewer than three example blocks before the next definition
- **THEN** the chapter is non-conforming and examples are added (a figure counts as an example only together with its explanation)

### Requirement: Bottom-up order within and across chapters
Every chapter SHALL introduce the components of a notion before the notion itself, and the physics part SHALL be
ordered classical mechanics, field objects (scalar, complex, vector, spinor fields), classical field theory, gauge
theory, quantum field theory, the Higgs mechanism, general relativity, supersymmetry, supergravity, each in its own
chapter.

#### Scenario: Field theory before fields
- **WHEN** a chapter uses a kind of field (a spinor, a complex scalar) in an action
- **THEN** that kind of field was defined, with examples, in an earlier chapter or section

#### Scenario: Mechanics before fields
- **WHEN** the Lagrangian formalism for fields is introduced
- **THEN** the chapter on classical mechanics (Lagrangian, Hamiltonian, Noether's theorem for finitely many degrees of freedom) precedes it

### Requirement: Chapters are small and single-topic
Each chapter SHALL treat one topic and introduce at most six definitions; a topic needing more is split into
consecutive chapters.

#### Scenario: A chapter grows past the limit
- **WHEN** the definitions checker counts more than six definition blocks in a chapter
- **THEN** the chapter is non-conforming and is split, each part keeping its own examples, figure, exercises and summary

#### Scenario: Manifolds are several chapters
- **WHEN** the mathematics part is listed
- **THEN** topological spaces, charts and manifolds, embeddings, tangent vectors and one-forms are separate chapters, and likewise for the other topics of the design's chapter map

### Requirement: Primitives are defined before the notions built on them
Every definition SHALL use only terms defined in an earlier chapter, an earlier section of the same chapter, or
a primer chapter, and the mathematics part SHALL open with primer chapters on sets and maps, topology, linear
algebra and calculus on `ℝ^n`.

#### Scenario: Manifold defined
- **WHEN** a manifold is defined
- **THEN** topological space, open set, continuous map, homeomorphism and Hausdorff space were defined in the topology primer, and vector space, basis, dual space and bilinear form in the linear algebra primer

#### Scenario: A primitive is missing
- **WHEN** a definition's body uses a term that no earlier definition introduced
- **THEN** the chapter is non-conforming and the term is defined before it, in a primer chapter if it is elementary

### Requirement: Every term is defined before it is used
Every mathematical or physical term the tutorial uses (for example embedding, Jacobian, Lie algebra, gauge
transformation, spinor, superfield) SHALL be defined in the text before its first use, or carry a cross-reference to
the earlier definition. The tutorial SHALL NOT rely on the reader's prior knowledge of such terms.

#### Scenario: A term appears without definition
- **WHEN** a term is used in a chapter and neither that chapter nor an earlier one defines it
- **THEN** the chapter is non-conforming and the definition is added at the first use

### Requirement: Chapters contain exercises with folded solutions
Each chapter SHALL contain exercises that ask the reader to compute something related to the chapter's results,
each followed by a solution that is collapsed by default and gives the derivation theoretically, as a Sage
computation, as a Lean statement and proof, or in all three forms.

#### Scenario: Reader attempts an exercise
- **WHEN** a reader reaches an exercise
- **THEN** the solution is hidden until the reader expands it, and when expanded it shows at least one complete solution form

#### Scenario: A Sage solution is given
- **WHEN** an exercise's solution includes Sage code
- **THEN** the code is a committed script whose recorded output matches the output shown in the solution

#### Scenario: A Lean solution is given
- **WHEN** an exercise's solution includes a Lean proof
- **THEN** the statement is a theorem of the library, shown through the same mechanism as the chapter's results

### Requirement: Abelian Kaluza-Klein chapter covers the complete reduction
The abelian part SHALL present the five-dimensional Einstein-Hilbert action, the metric ansatz with a scalar, a
vector and a four-dimensional metric, the reduction to gravity plus Maxwell plus a scalar, the geometric origin of
gauge transformations, the Kaluza-Klein mode expansion with its mass spectrum and the quantisation of electric
charge in terms of the circle radius.

#### Scenario: Reader follows the mass spectrum derivation
- **WHEN** the mode expansion of a five-dimensional massless scalar on the circle is performed
- **THEN** the four-dimensional masses `m_n = |n| / R` and charges `q_n` are derived with the relation between `R`, the gauge coupling and the Planck mass displayed

### Requirement: Non-abelian Kaluza-Klein chapter derives the gauge group from isometries
The non-abelian part SHALL show how the Killing vectors of `S^2` enter the metric ansatz, derive that the resulting
four-dimensional vector fields transform as `SU(2)` gauge fields, and identify the structure constants of the gauge
algebra with those of the isometry algebra.

#### Scenario: Gauge transformation from a coordinate transformation
- **WHEN** an internal coordinate transformation along a Killing vector is applied to the ansatz
- **THEN** the induced transformation of the four-dimensional vector fields is displayed and matches the non-abelian gauge transformation with the computed structure constants

### Requirement: CSDR chapter reaches an explicit four-dimensional theory
The CSDR part SHALL state the symmetry condition on gauge fields over `M_4 x S/R`, derive the constraints, solve
them for `S/R = SU(2)/U(1)` with a stated higher-dimensional gauge group, and display the four-dimensional gauge
group, the scalar field content, the Higgs potential and the resulting symmetry breaking.

#### Scenario: Solving the constraints for a chosen gauge group
- **WHEN** the higher-dimensional gauge group is `SU(3)` and `R = U(1)` is embedded in it
- **THEN** the four-dimensional gauge group is derived as the centraliser of the embedding, the scalars are read off from the branching of the adjoint, and the potential is computed and minimised explicitly

### Requirement: The survey covers every method of compactification and reduction in logical order
The survey part SHALL present every method of compactification and dimensional reduction in a logical order, from
toroidal reductions through group manifolds, supergravity sphere reductions, symmetric-field schemes and
special-holonomy and flux compactifications to warped and large extra dimensions. Each method SHALL state the
ansatz, the reduced content, and its relation to the methods before it.

#### Scenario: The survey's order
- **WHEN** the survey's table of contents is read
- **THEN** its chapters are, in order: toroidal and Scherk-Schwarz reductions; group-manifold reductions and the consistency problem of coset reductions; supergravity sphere reductions (Freund-Rubin, consistent truncations, exceptional field theory); the symmetric-field schemes compared (CSDR, Scherk-Schwarz, equivariant, fuzzy); special-holonomy compactifications (Calabi-Yau, `G_2`, nearly-Kähler, orbifolds); flux compactifications and moduli stabilisation; warped and large extra dimensions

#### Scenario: A method is looked up
- **WHEN** a reader looks for a compactification method in the survey
- **THEN** its chapter states the ansatz, the reduced content, the consistency status and the relation to the preceding methods, with citations to papers that were read

#### Scenario: Each method has a verified worked example
- **WHEN** a survey chapter presents a method
- **THEN** it contains at least one explicit computation (a spectrum, a field equation, a structure identity, a potential minimum, a warp factor) verified in Lean, or labelled with its non-formal level

### Requirement: Each chapter ends with a verified-results summary (REMOVED by design D29: the main text never refers to verification; the registry appendix lists every claim)
Each chapter SHALL end with a list of the results it established, each paired with the identifier of the Lean
theorem that verifies it or with the label of its non-formal verification level.

#### Scenario: Chapter summary consulted
- **WHEN** a reader reaches the end of a chapter
- **THEN** every displayed computational result of that chapter appears in the summary with its verification identifier

### Requirement: The foundation parts are named as foundations

The two foundation parts SHALL be titled "Foundations of mathematics" and "Foundations of physics".

#### Scenario: Table of contents
- **WHEN** the reader opens the table of contents
- **THEN** the first two parts are titled "Foundations of mathematics" and "Foundations of physics"

### Requirement: Chapter introductions are self-contained

Each chapter introduction SHALL motivate its topic on its own terms and SHALL NOT state the connection to
Kaluza-Klein theory or coset space dimensional reduction; that connection is stated in the part introduction and
where the object is used.

#### Scenario: Introduction read in isolation
- **WHEN** a reader reads the introduction of a foundation chapter
- **THEN** it explains what the objects are and what the reader will compute, names the sources, and does not
  mention the main theme of the lectures

### Requirement: Chapters are organised on two levels within a part

Within each part, chapters SHALL be thematic groups and their sections SHALL be the single-topic units of at most
six definitions; section identifiers SHALL remain number-free slugs.

#### Scenario: A group chapter
- **WHEN** the reader opens the chapter "Sets, relations and topology"
- **THEN** it contains the sections on sets and maps, relations and quotients, topological spaces, and compact and
  connected spaces, each with its own definitions, examples, exercises and figure

#### Scenario: Cross-references after regrouping
- **WHEN** chapters are regrouped
- **THEN** every existing cross-reference still resolves

### Requirement: Block kinds are colour coded

Statements (definitions, theorems, lemmas, propositions, corollaries) SHALL have a red title; examples SHALL have a
green title; exercises SHALL have a blue title; no block and no solution callout SHALL have a vertical bar or a
background highlight. The conventions chapter SHALL contain a legend of the three colours.

#### Scenario: a rendered chapter

- **WHEN** a chapter with a definition, a proposition, an example and an exercise is rendered to HTML
- **THEN** the definition and the proposition have red titles, the example a green title, the exercise a blue
  title, and none of them, nor the solution callout, a vertical bar or a coloured background.

### Requirement: Every definition, theorem and proposition has graded examples and exercises

Each definition, theorem and proposition SHALL be followed by at least three examples of increasing complexity
and at least three exercises of increasing difficulty with collapsed solutions; examples SHALL include a figure
or plot where the object can be drawn and an interactive figure where feasible. Three is a minimum, not a target: a
central notion, one with several standard cases, or one for which a textbook offers good problems SHALL get more
(design D21, D32).

#### Scenario: Definition inspected
- **WHEN** the checker inspects a `def-`, `thm-` or `prp-` block
- **THEN** at least three `exm-` blocks and at least three `exr-` blocks with solution callouts follow it before
  the next definition, theorem or proposition

#### Scenario: Interactive example
- **WHEN** an example draws a surface or a vector field
- **THEN** the HTML output embeds an interactive version and the PDF output the static image

### Requirement: Chapters end with integrative exercises

Each chapter SHALL end with three exercises combining material from several of its sections, with collapsed
solutions.

#### Scenario: Chapter end
- **WHEN** the checker reaches the end of a chapter file
- **THEN** it finds a section of three `exr-` blocks with solutions after the last definition section

### Requirement: The foundation parts are organised before they are written
The table of contents of the two foundation parts (design D24: group chapters and their sections, with the notions
each section defines and the computations it verifies) SHALL be fixed and scaffolded before any new section is
written: every group file SHALL exist with its introduction placeholder, its `##` section headings with their
identifiers and a one-line scope statement each, and the registry SHALL list the planned verified computations
with their recipes. New content SHALL be written group by group in the order of the table of contents.

#### Scenario: Scaffold precedes content
- **WHEN** a new foundation section is written
- **THEN** its group file, section heading and identifier already exist in `_quarto.yml` and in the design's table of contents, and the section's planned definitions and verified computations are the ones the design lists (or the design is updated first)

#### Scenario: A draft render shows the whole structure
- **WHEN** the book is draft-rendered after scaffolding
- **THEN** every group chapter and every section of design D24 appears in the table of contents, the unwritten ones marked "To be written" with their scope statement

### Requirement: Analysis, probability and operator theory are taught in the mathematics part
The mathematics part SHALL contain group chapters on real analysis (limits, series, continuity and
differentiability, the Riemann integral, calculus on `ℝⁿ`, multiple integrals and convergence of functions),
complex analysis (complex functions, holomorphy, contour integration, series and residues, analytic continuation and
the Gamma function), differential equations (ordinary differential equations, linear systems and the matrix
exponential, Sturm-Liouville problems, partial differential equations, Green's functions), Fourier analysis (Fourier
series, the Fourier transform, distributions, harmonic analysis on the sphere and on compact groups), Hilbert spaces
and operator theory (inner product and Hilbert spaces, bounded operators, spectral theory, unbounded operators and
the Laplacian, traces and heat kernels) and probability and statistics (probability spaces, random variables and
distributions, limit theorems, statistics and estimation), each in the format of the other groups and each with
its verified computations.

#### Scenario: A later chapter needs an analytic tool
- **WHEN** a physics chapter uses a Fourier mode expansion, a Gaussian integral, a residue, a spectral decomposition, an operator commutator or a probability distribution
- **THEN** the notion is defined in the corresponding mathematics group and the chapter cites that definition

#### Scenario: Kaluza-Klein spectrum as a spectral statement
- **WHEN** the mode expansion on the circle or the sphere is introduced in Part III or IV
- **THEN** it cites the Laplacian spectra `n²/R²` and `l(l+1)/R²` verified in the operator-theory group and the Fourier-series and spherical-harmonics sections

### Requirement: Classical mechanics, thermodynamics, quantum mechanics and quantum field theory are taught in the physics part
The physics part SHALL contain group chapters on classical mechanics (Newtonian, Lagrangian and Hamiltonian
mechanics, Poisson brackets and canonical transformations, Noether's theorem, small oscillations), thermodynamics
and statistical mechanics (the laws and potentials, ensembles and partition functions, simple systems, quantum
statistics, phase transitions and Landau theory, finite temperature as a compact Euclidean time), quantum mechanics
(states and observables, measurement and the Schrödinger equation, the harmonic oscillator, angular momentum and
spin, symmetries, perturbation theory and the path integral) and quantum field theory (canonical quantisation of the
free scalar, propagators and Green's functions, the path integral and Euclidean field theory, perturbation theory
and Feynman diagrams, renormalisation, effective actions and anomalies in outline), each with its verified
computations and with the outline-level material labelled `text` with citations.

#### Scenario: Quantum mechanics builds on the operator-theory group
- **WHEN** the quantum mechanics chapter defines observables, spectra or the harmonic-oscillator ladder operators
- **THEN** it cites the Hilbert-space, spectral-theorem and commutator definitions of the mathematics part and verifies the ladder-operator algebra with the `operator_identity` or `physlib/ladder_operators` recipe

#### Scenario: Thermal circle and Kaluza-Klein circle
- **WHEN** finite temperature is introduced in the statistical-mechanics chapter
- **THEN** the compact Euclidean time direction and its Matsubara modes are defined, and Part III cites them when it introduces the Kaluza-Klein circle and its modes

### Requirement: Blocks are connected by prose
Every statement, example and exercise block SHALL be preceded, within its section, by at least two sentences
(twenty words or more) of running text that introduce it; two blocks SHALL never follow each other directly
(solution callouts excepted, as they belong to their exercise). The format checker SHALL enforce the rule.

#### Scenario: A stacked pair of blocks
- **WHEN** a chapter contains an example block that starts right after the closing of the previous block, with no
  prose between them
- **THEN** `scripts/check_format.py` reports the pair and the chapter is non-conforming

#### Scenario: Reading the prose alone
- **WHEN** a reader reads only the running text of a section, skipping the blocks
- **THEN** the text still states what each block defines, illustrates or asks, and why it comes at that point

### Requirement: Functional analysis covers Banach spaces, not only Hilbert spaces
The Part I chapter on operators SHALL be a functional-analysis chapter: normed and Banach spaces with `ℓ^p`, `L^p` and
`C(K)` and their duals, the Hahn-Banach, uniform-boundedness, open-mapping and closed-graph theorems with
counterexamples, weak topologies and reflexivity, Sobolev spaces, compact and Fredholm operators with the Fredholm
index, and Banach and C*-algebras in outline, before and around the Hilbert-space material (design D31).

#### Scenario: The Fredholm index
- **WHEN** a later chapter uses the index of the Dirac operator
- **THEN** the Fredholm index and its stability are defined and proved earlier, in the functional-analysis chapter

#### Scenario: A theorem's hypothesis
- **WHEN** the chapter states the open-mapping or the uniform-boundedness theorem
- **THEN** an example shows the conclusion failing when completeness is dropped

### Requirement: The Dirac operator is analysed and solved
The book SHALL teach the Dirac operator in three places: a Part I chapter on spinors and the Dirac operator (Clifford
algebras, spin groups, spinor representations in every dimension and signature, spin structures, the spin
connection, the Dirac operator and its spectrum on tori and spheres, the Lichnerowicz formula, zero modes and the
index), a Part II chapter on the Dirac equation (free solutions, non-relativistic limit, the Dirac hydrogen atom
solved exactly, Foldy-Wouthuysen, charge conjugation, Weyl and Majorana fermions), and the higher-dimensional Dirac
operator on compact spaces: its splitting on `M^4 × K`, its spectrum and zero modes on tori, spheres in every
dimension, group manifolds, coset spaces `G/H`, Kähler and Calabi-Yau spaces and orbifolds, twisted by gauge
backgrounds, with higher-dimensional index theorems and the chirality problem (Witten's no-go and its evasion in
CSDR) (design D31).

#### Scenario: The spectrum on the sphere
- **WHEN** the reader studies the Dirac operator on `S^2`
- **THEN** its eigenvalues and eigenspinors are derived step by step, and an exercise computes them with Sage

#### Scenario: A coset space
- **WHEN** the reader studies the Dirac operator on a coset space `G/H` used in Part V or VI
- **THEN** its spectrum is derived by harmonic analysis on `G/H` (Frobenius reciprocity, Parthasarathy's formula on
  symmetric spaces), and the four-dimensional masses it gives are listed

#### Scenario: The chirality problem
- **WHEN** a chapter reduces a fermion over a positively curved space without gauge background
- **THEN** it shows that no chiral zero mode survives (Lichnerowicz, Witten), and the following section shows how a
  gauge background changes the index

#### Scenario: Chiral fermions from a reduction
- **WHEN** Part V reduces a fermion over `S^2` with a monopole background
- **THEN** the number of four-dimensional chiral zero modes is derived and equals the index computed in Part I

### Requirement: Renormalisation is taught extensively
The book SHALL have a Part II chapter on renormalisation covering regularisation schemes compared on the same
integrals, power counting, counterterms and renormalised perturbation theory at one and two loops, renormalisation
schemes, BPHZ with worked forests, Ward identities and QED at one loop, the renormalisation-group equations, beta
functions with asymptotic freedom, Dyson-Schwinger equations, the Wilsonian renormalisation group with the `ε`
expansion and critical exponents, effective field theories and decoupling, and renormalisation in higher dimensions
(Kaluza-Klein towers in loops, power-law running, threshold corrections, the Casimir energy) (design D31).

#### Scenario: A beta function
- **WHEN** the chapter derives the one-loop beta function of `φ⁴` theory
- **THEN** every step from the divergent integral to the beta function is shown, in two regularisation schemes that
  agree on the scheme-independent coefficient

#### Scenario: Critical exponents
- **WHEN** the chapter computes critical exponents in the `ε` expansion
- **THEN** it compares them with the exact two-dimensional Ising values of the statistical-mechanics chapter

### Requirement: Textbooks supply content, examples and exercises
Every textbook in `references/textbooks/` SHALL be read and mapped to the chapters it serves, with a reading note
naming the sections and problems used; content following a textbook SHALL be rewritten in the book's voice,
notation and conventions, and an adapted example or exercise SHALL cite the textbook with its problem number
(design D32).

#### Scenario: An adapted exercise
- **WHEN** an exercise is adapted from a textbook problem
- **THEN** it cites the textbook as `[@key, Problem n]` and the reading note lists that problem

#### Scenario: An unused textbook
- **WHEN** a textbook in the folder serves no chapter
- **THEN** `references/notes/README.md` records it with the reason

### Requirement: Special relativity has its own chapter before general relativity
The physics part SHALL contain a chapter on special relativity placed immediately before the general-relativity
chapter, covering the postulates and the Lorentz transformation, Minkowski spacetime and its causal structure, the
Lorentz and Poincaré groups treated in detail and formally, progressively from one-dimensional boosts to Wigner's
classification, relativistic kinematics and optics, relativistic dynamics
with accelerated observers, and electrodynamics in covariant form; earlier chapters that use special relativity SHALL
cross-reference it (design D33).

#### Scenario: Reading order
- **WHEN** the reader reaches the general-relativity chapter
- **THEN** the special-relativity chapter immediately precedes it in the sidebar, and the general-relativity chapter
  cites it for Minkowski spacetime and the Rindler frame

#### Scenario: Progressive formal treatment of the groups
- **WHEN** the reader works through the Lorentz and Poincaré groups
- **THEN** the chapter goes from boosts in one dimension to the formal definitions with proofs: the four components of
  `O(1, 3)`, the polar decomposition, `so(1, 3)` and its complexification, `SL(2, ℂ)` as double cover, the
  Thomas-Wigner rotation, the Poincaré group with its Casimirs, and Wigner's classification by little groups

#### Scenario: Problems from the problem book
- **WHEN** an exercise of the chapter is adapted from Blennow and Ohlsson
- **THEN** it cites the book and problem number, and the reading note lists the problem

