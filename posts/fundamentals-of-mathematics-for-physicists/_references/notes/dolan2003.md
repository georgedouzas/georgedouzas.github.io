# Identity

- arXiv: hep-th/0304037 (v1, 4 Apr 2003); preprint DIAS-STP-03-02.
- Title: The Spectrum of the Dirac Operator on Coset Spaces with Homogeneous Gauge Fields.
- Author: Brian P. Dolan (NUI Maynooth; DIAS Dublin). Journal version JHEP 05 (2003) 018 (external knowledge). 25 PDF pages.

# Structure

1. Introduction (higher-dimensional quantum Hall effect; square of the Dirac operator as a Hamiltonian, eq. (1)) [read].
2. Symmetric spaces (eqs. (2)-(16); examples i) S^2, eqs. (17)-(35); ii) S^4, eqs. (36)-(47); iii) CP^2, eqs. (48)-(58)) [read].
3. Non-symmetric spaces (torsion, eqs. (59)-(67); SU(3)/U(1)xU(1), eqs. (68)-(73)) [read]. 4. Conclusions [read].
Appendices A (geometry of cosets), B (CP^2 spectrum), C (index on SU(3)/U(1)^2) [skimmed].

# Conventions and mapping to the tutorial

- Euclidean gamma matrices {gamma^a, gamma^b} = 2 delta^ab; iD-slash = i gamma^a(e_a + (1/4) omega_ab gamma^ab + i A^i t_i) with
  Hermitian gauge generators t_i and real field strength F^i; Lichnerowicz (i D-slash)^2 = Delta + R/4 - (i/2) F gamma (eq. (5)).
  The tutorial uses anti-Hermitian potentials A = i a; with F = i da its formula reads D^2 = nabla*nabla + R/4 - (1/2) F_ab gamma^a
  gamma^b (checked in exr-spd-lichnerowicz-check).
- Chirality counted with sigma_3 (gamma^1 = sigma_1, gamma^2 = sigma_2): his index nu = -M (eq. (32)) is +M with the tutorial's
  gamma_* = -sigma_3 (exr-spd-index-sign).
- Casimirs C_2(SU(2), j) = j(j + 1); unit spheres, R(S^2) = 2, R(S^4) = 12.

# Sections and equations used (part-i-13-spinors-dirac-operator)

- Eqs. (2), (4), (5): def-dirac-operator, prp-spinor-curvature, thm-lichnerowicz, exm-spd-lichnerowicz-twisted,
  def-twisted-dirac-operator. Remark after (26) (no zero modes for M = 0 by Lichnerowicz): thm-lichnerowicz-vanishing,
  thm-chirality-no-go.
- Eqs. (15), (16), (67): thm-parthasarathy, exr-spd-casimir-laplacian; App. A: def-homogeneous-spinor-bundle.
- S^2, eqs. (17)-(32): exm-spd-monopole-potential (charge-2 spin connection, eqs. (19)-(21)), thm-monopole-index (eqs. (28)-(32)),
  exm-spd-landau (Sec. 1 and eqs. (29)-(31)), exr-spd-monopole-casimirs (eqs. (27)-(31)), exr-spd-index-sign (eq. (32)),
  exr-spd-zero-mode-balance and exm-spd-spin-connection-background (remark after (32): |M| = 1 gives a constant spinor).
- S^4, eqs. (36)-(47): exr-spd-s4-casimirs (eqs. (36)-(43)), exm-spd-instanton, exr-spd-instanton-check (eq. (44)),
  exm-spd-ch-instanton (eq. (46)), exm-spd-as-four, exr-spd-s4-instanton-modes (remark after (47)), exr-spd-instanton-chirality.
- CP^2, Sec. 2 iii, eqs. (48)-(58): thm-spin-obstruction, exm-spd-cp2-not-spin, def-spinc-structure, exm-spd-spinc-cp2,
  exm-spd-cp2-homogeneous, exm-spd-dolbeault-cp2 (eq. (56)), exr-spd-cp2-index-table and figure sd-cp2-index (eqs. (55)-(56)),
  exr-spd-cp2-two-views, exr-spd-cp2-chirality.
- Sec. 3, eqs. (59)-(73): exm-spd-nearly-kahler (eqs. (59)-(67), (70)), exr-spd-flag-restriction (eq. (69)), exr-spd-flag-index
  (eqs. (70)-(72)).
- prp-chiral-zero-modes-gauge cites eqs. (16), (31)-(32), (44)-(47).
