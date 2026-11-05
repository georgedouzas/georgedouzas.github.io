# Identity

- arXiv: hep-th/0611263 (v2, 13 June 2021).
- Title: The unitary representations of the Poincaré group in any spacetime dimension.
- Authors: Xavier Bekaert and Nicolas Boulanger. Lecture notes, 62 pages.

# Structure

1. Group-theoretical preliminaries (pp. 3-6): universal cover of the Lorentz group (Sec. 1.1), the Poincaré group and
algebra with brackets eqs. (1)-(3) and Casimirs eqs. (4)-(6), the Pauli-Lubanski vector for $D = 4$ (Sec. 1.2), finite-dimensional
unitary representations of non-compact simple groups are trivial (Sec. 1.3).
2. Elementary particles as unitary irreducible representations (pp. 6-9), after Bargmann and Wigner.
3. Classification (pp. 9-16): induced representations eqs. (9)-(13) (Sec. 3.1), orbits and stability subgroups with the
algebraic proof that the little group of a light-like momentum is $ISO(D - 2)$, eqs. (14)-(17) (Sec. 3.2), the classification
table and the helicity / infinite-spin split (Sec. 3.3), corollary on infinite dimensionality.
4. Young diagrams (pp. 16-35). 5. Relativistic field equations (pp. 35-51). Appendices A (Siegel-Zwiebach) and B (three
dimensions: anyons, fractional spin).

# Conventions

- Signature mostly plus $(-, +, \dots, +)$; our $(+, -, -, -)$ is obtained by $\eta \to -\eta$, which changes the sign of $p^2$ in their
  orbit table ($p^2 = -m^2$ there is $p^2 = m^2$ here) and of the quadratic Casimir $C_2 = -P^\mu P_\mu$.
- Hermitian generators with $i[M, M] = \dots$, eq. (1), and translations $U(1, a) = \exp(-iP^\mu a_\mu)$ in their signature. The chapter
  uses $\hat P^\mu = -i\,dU(P^\mu)$, $\hat M^{\mu\nu} = i\,dU(M^{\mu\nu})$, $U(1, a) = e^{ia_\mu\hat P^\mu}$ in $(+, -, -, -)$, chosen so that $e^{-ip\cdot x}$ has momentum $p$
  and rotations act as $e^{-i\theta\hat J}$ (prp-sr-poincare-algebra).

# Sections, equations and problems used (part-ii-08b-special-relativity, task 12.9)

- Sec. 1.1 (double cover, $SO^+(1, 3)$ connected but not simply connected): thm-sr-double-cover, exr-sr-sl2c-simply-connected.
- Sec. 1.2, eqs. (1)-(6) (Poincaré algebra, Casimirs, Pauli-Lubanski vector): def-sr-poincare-group, prp-sr-poincare-algebra,
  def-sr-pauli-lubanski, thm-sr-casimirs.
- Sec. 1.3 (non-trivial finite-dimensional unitary representations of non-compact simple groups are trivial):
  exr-sr-field-complex, thm-sr-wigner-classification part 4.
- Sec. 3.1, eqs. (9)-(13) (induced representations, little-group element $W(\Lambda, p) = L^{-1}(\Lambda p)\Lambda L(p)$): thm-sr-induced; Mackey's
  completeness and irreducibility are cited from this section in thm-sr-wigner-classification.
- Sec. 3.2 (orbits; light-like little group $ISO(D - 2)$ from the brackets, eqs. (14)-(16)): prp-sr-little-groups, def-sr-little-group.
- Sec. 3.3 (classification table, helicity and continuous-spin representations): thm-sr-wigner-classification.
- App. B (three dimensions, fractional spin): exr-sr-helicity-quantised.
