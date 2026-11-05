# Identity

- arXiv: hep-th/9910030 (v7, 12 Feb 2016); preprint KUL-TF-99/37.
- Title: Tools for supersymmetry. Author: Antoine Van Proeyen (K.U. Leuven).
- Lectures at the spring school "Q.F.T., Supersymmetry and Superstrings" (Calimanesti, 1998). 50 PDF pages; section numbers
  are those of the text.

# Structure

1. Introduction. 2. Bosonic spacetime symmetries (2.1-2.5). 3. Spinors in arbitrary dimensions: 3.1 Gamma matrices and their
symmetry; 3.2 Irreducible spinors; 3.3 4 dimensions; 3.4 Technical tips [read in full]. 4. Supersymmetry algebras (4.1-4.7)
[read 4.3 with Table 3]. Appendices (not used).

# Conventions and mapping to the tutorial

- Signature: d = t + s with eta = diag(-...-, +...+), timelike first and NEGATIVE (eq. (3.1)). The tutorial's signature (p, q)
  counts p directions of square +1; for Lorentzian spacetime (+,-,-,-) the roles of time and space are exchanged. The
  classification depends only on s - t mod 8 for the Majorana condition with both signs of eta allowed in even dimension,
  and the set {0, 1, 2, 6, 7} is invariant under s - t -> t - s, so the tutorial states it with p - q.
- Gamma matrices by tensor products of Pauli matrices, eq. (3.3); hermiticity (3.4)-(3.5); chirality Gamma_* =
  (-i)^{d/2+t} Gamma_1 ... Gamma_d (3.8), which differs in sign from the tutorial's i^{d(d-1)/2+q} gamma^1...gamma^d
  (noted in exm-spd-chirality-two).
- Charge conjugation C^T = -epsilon C, Gamma^T = -eta C Gamma C^{-1} (3.11); Table 1 lists (epsilon, eta) by d mod 8 and is
  independent of the signature.

# Sections, equations and tables used (part-i-13-spinors-dirac-operator)

- Sec. 3.1, eqs. (3.3)-(3.5): def-gamma-matrices-general; after eq. (3.10) (Gamma_* as the next gamma matrix in odd
  dimension): exm-spd-gammas-eleven; eq. (3.8): exm-spd-chirality-two; footnote 5 (two inequivalent representations in odd
  dimension): thm-pauli-uniqueness.
- Eqs. (3.11)-(3.16) and Table 1: def-charge-conjugation-matrix, exm-spd-charge-two, exm-spd-charge-eleven,
  exr-spd-charge-table, exr-spd-symmetric-count (counting of symmetric C Gamma^(n), eq. (3.16)).
- Sec. 3.2, eqs. (3.19)-(3.28): def-majorana-condition, exr-spd-majorana-invariant (eq. (3.23)), exr-spd-mw-condition (eq.
  (3.28)), thm-majorana-classification (eqs. (3.20), (3.25), (3.28)), prp-weyl-conjugation (paragraph before (3.28)).
- Table 2 (possible spinors and minimal real dimension by d and t mod 4): thm-majorana-classification, exm-spd-lorentz-table
  (column t = 1), checked against figure sd-majorana-table.
- Sec. 4.3 with Table 3 (pure supergravities with Q = 32, 16, 8, 4; the limit of 32 supercharges from helicities <= 2):
  exm-spd-ten-eleven, exm-spd-weyl-ten.
- Eqs. (3.3), (3.8): prp-spinor-branching (product construction of gamma matrices).
