# Identity

- arXiv: 0708.3222 [hep-th] (v1, 23 Aug 2007); preprint number CERN-PH-TH 2007/144.
- Title: "On the Consistency of Coset Space Dimensional Reduction".
- Authors: A. Chatzistavrakidis, P. Manousselis, N. Prezas **and G. Zoupanos** (the PDF lists four authors; the task brief named only the first three).
- Year: 2007. Venue: not stated in the PDF (called "this letter" in the text). Length: 14 PDF pages (title page unnumbered, then printed pages 1-13).

Page convention below: "PDF p.N" is what the Read tool shows; printed page = N-1.

# Structure

1. Abstract -- PDF p.1 (title page, no printed number).
2. Section 1, Introduction -- PDF p.2-4 (printed 1-3). Contains the commutative diagram and the definition of a consistent truncation (PDF p.3-4).
3. Section 2, Reduction on coset spaces: brief reminder -- PDF p.4-6 (printed 3-5).
   - 2.1 Geometry of coset spaces -- PDF p.4-5, eqs. (2.1)-(2.12).
   - 2.2 Reduction on group manifolds and coset spaces -- PDF p.5-6, eqs. (2.13)-(2.18).
4. Section 3, CSDR of Yang-Mills theory -- PDF p.6-11 (printed 5-10).
   - 3.1 Gravity background -- PDF p.6-7, eqs. (3.1)-(3.6).
   - 3.2 Reduction of the action -- PDF p.7-9, eqs. (3.7)-(3.21).
   - 3.3 Four-dimensional gauge group and spectrum -- PDF p.9-10, eqs. (3.22)-(3.23).
   - 3.4 Reduction of the equations of motion -- PDF p.10-11, eqs. (3.24)-(3.32).
   - 3.5 Bianchi identities -- PDF p.11, eqs. (3.33)-(3.37).
5. Section 4, Conclusions -- PDF p.12 (printed 11).
6. Acknowledgements, References [1]-[24] -- PDF p.12-14 (printed 11-13).

# Conventions

- **Signature / spacetime**: $M^D = M_4 \times S/R$, $M_4$ four-dimensional Minkowski with flat metric $\eta_{mn}$ (sign convention not stated explicitly); $S/R$ compact coset of dimension $d = D-4$. Differential-form language throughout (wedge products, Hodge stars $*_D$, $*_4$, $*_d$, volume forms $vol_4$, $vol_d$).
- **Index conventions** (footnote 3, PDF p.7): tangent spacetime $m,n,\dots$, curved spacetime $\mu,\nu,\dots$; tangent coset $a,b,\dots$, curved coset $\alpha,\beta,\dots$; full group $S$ tangent indices $I,J,\dots$ and curved $A,B,\dots$ (but in Sec. 2 the capital $A,B,C$ are used as Lie-algebra indices of $S$ and $I$ for Killing vectors); subgroup $R$ indices $i,j,\dots$; gauge-algebra indices of $G$ are $\tilde I,\tilde J,\tilde K = 1,\dots,\dim G$. Coordinates $x^\mu$ (spacetime), $y^\alpha$ (coset). Hats denote $D$-dimensional fields. Tangent coset indices are raised/lowered with $\delta_{ab}$, so upper and lower $a,b$ are equivalent (PDF p.5).
- **Coset geometry** (Sec. 2.1): group coordinates $(y^a, z^i)$, $s \sim e^{y^a Q_a} e^{z^i Q_i}$, coset representative $L(y) = e^{y^a Q_a}$. Maurer-Cartan form $e(y) = L^{-1}(y)\,dL = e^A Q_A = e^a Q_a + e^i Q_i$ (2.1); $e^a$ is the coframe, $e^i = e^i_a(y) e^a$ the $R$-connection. $de = -e\wedge e = -[e,e]$ (2.2), $de^A = -\tfrac12 f^A_{BC} e^B \wedge e^C$ (2.3).
- **Reductive algebra** (2.4), PDF p.5:
  $[Q_i,Q_j] = f^k_{ij} Q_k,\quad [Q_i,Q_a] = f^b_{ia} Q_b,\quad [Q_a,Q_b] = f^c_{ab} Q_c + f^i_{ab} Q_i$, implying $f^j_{bi}=0$. (Structure-constant placement: $f^{\text{upper}}_{\text{lower lower}}$, e.g. $f^b_{ia}$; later also written $f_{ib}{}^c$, $f_{ij}{}^k$ in (3.19)-(3.20).) For the consistency proof they further assume $S$ unimodular ($f^A_{AB}=0$), $f^a_{ab}=0$, and $f_{ABC}$ completely antisymmetric (PDF p.10-11).
- **Metric** on the coset: $S$-invariant metric $g(y) = \gamma_{ab} e^a(y) e^b(y)$ (2.12), $\gamma_{ab}$ the Killing metric of $S$ restricted to $S/R$. No radii $R_a$ are introduced anywhere; the internal metric is fixed and non-dynamical. Full background metric $g_{(D)} = \eta_{mn} e^m e^n + \gamma_{ab} e^a e^b$ (3.4). Killing vectors of the left isometry $S$: $X_I$.
- **Gauge field / field strength**: $\hat F_{(2)} = d\hat A_{(1)} + \hat A_{(1)}\wedge \hat A_{(1)}$ (unnumbered, PDF p.7), Lie-algebra valued in $G$; in components $\hat F^{\tilde I} = d\hat A^{\tilde I} + \tfrac12 f^{\tilde I}_{\tilde J\tilde K}\hat A^{\tilde J}\wedge\hat A^{\tilde K}$ (3.10). **No explicit gauge coupling $g$ appears** (absorbed into $A$); no normalisation of $\mathrm{Tr}$ is stated.
- **Higher-dimensional action**: Einstein-Yang-Mills with cosmological constant,
  $\mathcal L = \hat R\, *_D \mathbf 1 - \tfrac12 \mathrm{Tr}\,\hat F_{(2)}\wedge *_D\hat F_{(2)} - \lambda_{(D)} *_D \mathbf 1$ (3.1), PDF p.6; pure YM part $S = -\tfrac12\int \mathrm{Tr}\,\hat F\wedge *_D \hat F$ (3.15), PDF p.8.

# Equations the tutorial uses

## (a) CSDR symmetry condition and derivation of the constraints

- Scherk-Schwarz ansatz on a group manifold $S$ (2.13), PDF p.5: $A = A_\mu dx^\mu + A_I(x) e^I(y)$, $I=1,\dots,\dim S$, $e^I$ left-invariant 1-forms; keeps only $G_L$ singlets of $G_L\times G_R$. Invariance condition (2.14): $\mathcal L_{X^I} A = 0$, $X^I$ Killing vectors dual to the right-invariant 1-forms (footnote 2: right-invariant vector fields generate left translations). Metric: $\mathcal L_{X^I} g_{MN} = 0$ (2.15).
- CSDR generalised invariance condition (2.16), PDF p.6: $\mathcal L_{X^I} A = D W_I$, $W_I$ a gauge parameter associated to $X_I$. Expanded (2.17): $\mathcal L_{X^I}A = i_{X^I} dA + d\, i_{X^I} A = D W_I = dW_I + [A, W]$, together with the consistency condition (2.18): $[\mathcal L_{X^I},\mathcal L_{X^J}] = \mathcal L_{[X^I,X^J]}$. The paper does NOT re-derive the solution of (2.17)-(2.18); it refers to Forgacs-Manton [13] and Kapetanakis-Zoupanos [14].
- Reduction ansatz (3.7)-(3.9), PDF p.7-8: $\hat A^{\tilde I}(x,y) = A^{\tilde I}(x) + \chi^{\tilde I}_\alpha(x,y)\,dy^\alpha$, $\chi^{\tilde I}_\alpha(x,y) = \phi^{\tilde I}_A(x)\, e^A_\alpha(y)$, i.e. $\hat A^{\tilde I} = A^{\tilde I} + \phi^{\tilde I}_A e^A$, with $A$ running over ALL of $\mathrm{Lie}(S)$ (both $e^a$ and $e^i$). "The ansatz solves the generalized symmetry condition (2.16) provided we can embed $R$ in $G$."
- Field strength from the ansatz, using (2.3), (3.11)-(3.14), PDF p.8:
  $\hat F^{\tilde I} = F^{\tilde I} + D\phi^{\tilde I}_A\wedge e^A - \tfrac12 F^{\tilde I}_{AB} e^A\wedge e^B$,
  $F^{\tilde I} = dA^{\tilde I} + \tfrac12 f^{\tilde I}_{\tilde J\tilde K} A^{\tilde J}\wedge A^{\tilde K}$ (3.12),
  $D\phi^{\tilde I}_A = d\phi^{\tilde I}_A + f^{\tilde I}_{\tilde J\tilde K} A^{\tilde J}\phi^{\tilde K}_A$ (3.13),
  $F^{\tilde I}_{AB} = f^C_{AB}\phi^{\tilde I}_C - [\phi_A,\phi_B]^{\tilde I}$ (3.14).
  Remark (PDF p.8): using the $e^A$ (not only $e^a$) basis is what gives constant $f^A_{BC}$ instead of $y$-dependent $C^a_{bc}(y)$ of (2.7), $C^a_{bc} = f^a_{bc} - 2 e^i_{[b} f^a_{c]i}$.
- Hodge dual (3.16), PDF p.8: $*_D\hat F^{\tilde I} = *_4 F^{\tilde I}\wedge vol_d + *_4 D\phi^{\tilde I}_A\wedge *_d e^A - \tfrac12 F_{AB}\, vol_4\wedge *_d(e^A\wedge e^B)$.
- Inserting (3.11), (3.16) into (3.15) gives the 4D Lagrangian (3.17), PDF p.9:
  $\mathcal L = -\tfrac12 \mathrm{Tr}\, F\wedge *_4 F + \tfrac12 \mathrm{Tr}\, D\phi_a\wedge *_4 D\phi_a - \tfrac14 F_{ab}F^{ab}\, vol_4$,
  "provided we impose the CSDR constraints" (3.18): $D\phi^{\tilde I}_i = F^{\tilde I}_{ai} = F^{\tilde I}_{ij} = 0$. Explicitly (3.19)-(3.21):
  $F_{ib} = f_{ib}{}^c\phi_c - [\phi_i,\phi_b] = 0$, $F_{ij} = f_{ij}{}^k\phi_k - [\phi_i,\phi_j] = 0$, $[A_\mu,\phi_i] = 0$.
  (3.21) comes from $D\phi_i = 0$ with constant $\phi_i$ (Lorentz invariance). $F_{ab}$ in (3.17) depends on the $\phi_i$, which are set to the constant values fixed by (3.20).
  Note: the paper does not show intermediate algebra between (3.16) and (3.17); the constraints appear as the conditions needed for the cross terms with $e^i$ to drop out, not as a derivation from (2.17).
- Coset identities used (2.8)-(2.11), PDF p.5: $e^a\wedge *_d e^b = \delta^{ab} vol_d$; $d*_d e^a = C^b_{ba}(y)\, vol_d$; $e^a\wedge *_d(e^b\wedge e^c) = \delta^{ac} *_d e^b - \delta^{ab} *_d e^c$; $d*_d(e^a\wedge e^b) = (*e^a)C^c_{cb} - (*e^b)C^c_{ca} - C^c_{ab}(*e^c)$. Also (2.5)-(2.7): $de^a = -\tfrac12 f^a_{bc}e^b\wedge e^c - f^a_{bi} e^b\wedge e^i$, $de^i = -\tfrac12 f^i_{ab} e^a\wedge e^b - \tfrac12 f^i_{jk} e^j\wedge e^k$, $de^a = -\tfrac12 C^a_{bc}(y) e^b\wedge e^c$.

## (b) Solution of the constraints: 4D gauge group, scalars, potential

- Sec. 3.3, PDF p.9-10. (3.20) implies the $\phi_i$ are generators of $R$ (assumed a Lie subgroup of $G$); (3.21) then says $A_\mu$ commutes with the embedded $R$: 4D gauge group $H = C_G(R)$ (centraliser of $R$ in $G$).
- Scalars $\phi_a(x)$: from (3.19) they transform under $R$ as the vector $v$ of (3.22), $S\supset R$, $\mathrm{adj}S = \mathrm{adj}R + v$, and are intertwining operators; by Schur's lemma, with (3.23) $G\supset R$, $\mathrm{adj}G = (\mathrm{adj}R,1) + (1,\mathrm{adj}H) + \sum(r_i,h_i)$ and $v = \sum s_i$, one obtains an $h_i$ multiplet of scalars for every pair $(r_i,s_i)$ with $r_i\simeq s_i$ (PDF p.10).
- Potential: not written as a separate formula; it is the term $-\tfrac14 F_{ab}F^{ab}\,vol_4$ in (3.17) with $F_{ab} = f^C_{ab}\phi_C - [\phi_a,\phi_b] = f^c_{ab}\phi_c + f^i_{ab}\phi_i - [\phi_a,\phi_b]$ (from (3.14); the split form is written on PDF p.11 just after (3.32)). No explicit formula for $V(\phi)$ with a trace and sign is isolated; the tutorial should derive it from (3.14) and (3.17).

## (c) Consistency: definition, criterion, result

- Two kinds of truncation (PDF p.3): (a) ordinary KK reduction keeping only massless modes, (b) truncation by constraints that reduce the number of independent field components. Both are truncations; consistency means solutions of the e.o.m. of $\mathcal L_R$ are solutions of the e.o.m. of $\mathcal L$: commutative diagram $\mathcal L \to \mathcal L_R$, $\frac{\delta\mathcal L}{\delta\Phi}=0 \to \left(\frac{\delta\mathcal L}{\delta\Phi}\right)_R = 0 \Leftrightarrow \frac{\delta\mathcal L_R}{\delta\Phi}=0$ (PDF p.3).
- Definition (PDF p.4, italic): "a truncation is said to be consistent when its implementation at the level of the variational principle agrees with its implementation at the level of the equations of motion", i.e. truncating the Lagrangian then varying commutes with varying then truncating.
- Criterion (PDF p.4, following Pons-Talavera [20] and Pons [7]): for constraint-driven truncations a Dirac-Bergmann-like mechanism applies; the appearance of **secondary constraints** (dynamically derived from the primary ones) is the typical obstruction. Claim: CSDR is a truncation of the second type and is consistent; "no new constraints are needed for the reduction of the higher-dimensional equations of motion besides the ones necessary for the proper reduction of the Lagrangian".
- Proof, Sec. 3.4, PDF p.10-11. YM equation (3.24): $\hat D *_D\hat F^{\tilde I} = \hat d *_D\hat F^{\tilde I} + f^{\tilde I}_{\tilde J\tilde K}\hat A^{\tilde J}\wedge *_D\hat F^{\tilde K} = 0$. Substituting (3.9), (3.16) gives the $vol_d$ component (3.25):
  $D*_4F^{\tilde I}\wedge vol_d - f^{\tilde I}_{\tilde J\tilde K}\phi^{\tilde J}_B *_4 D\phi^{\tilde K}_A\wedge e^A\wedge *_d e^B - *_4 D\phi^{\tilde I}_A\wedge d*_d e^A = 0$,
  and the $*_d e$ component (3.26):
  $D*_4 D\phi^{\tilde I}_A\wedge *_d e^A - \tfrac12 f^{\tilde I}_{\tilde J\tilde K}\phi^{\tilde J}_A F^{\tilde K}_{BC}\, vol_4\wedge e^A\wedge *_d(e^B\wedge e^C) - \tfrac12 F^{\tilde I}_{BC}\, vol_4\wedge d*_d(e^B\wedge e^C) = 0$.
  Imposing the same constraints (3.18) yields the 4D e.o.m. (3.27)-(3.28):
  $D*_4F^{\tilde I} = f^{\tilde I}_{\tilde J\tilde K}\phi^{\tilde J}_a *_4 D\phi^{\tilde K}_a$,
  $D*_4 D\phi^{\tilde I}_a = -\big(f^{\tilde I}_{\tilde J\tilde K}\phi^{\tilde J}_c F^{\tilde K}_{ca} + \tfrac12 F_{bc} f_{bca}\big) vol_4$,
  which are exactly the e.o.m. of (3.17): "Therefore the consistency of the truncation is verified."
- Key steps: the extra terms (3.29) $-f^{\tilde I}_{\tilde J\tilde K}\phi^{\tilde J}_i *_4 D\phi^{\tilde K}_b\wedge e^i_a e^a\wedge *_d e^b - *_4 D\phi^{\tilde I}_a\wedge d*_d e^a$ cancel because, for $S$ unimodular ($f^A_{AB}=0$) and reductive coset ($f^a_{ab}=0$), (2.7) and (2.9) give $d*_d e^a = -e^i_b f^b_{ai}\, vol_d$, so (3.29) becomes (3.30): $*_4\big(f^b_{ai}D\phi^{\tilde I}_a - f^{\tilde I}_{\tilde J\tilde K}\phi^{\tilde J}_i D\phi^{\tilde K}_b\big)e^i_b\wedge vol_d = *_4(DF^I_{ib})\, e^i_b\wedge vol_d = 0$, using $D\phi_i = 0$ and $F_{ib}=0$. For (3.28), the terms (3.31) $-\tfrac12 f^{\tilde I}_{\tilde J\tilde K}\phi^{\tilde J}_i F^{\tilde K}_{bc} vol_4\wedge e^i\wedge *_d(e^b\wedge e^c) - \tfrac12 F^{\tilde I}_{bc} vol_4\wedge d*_d(e^b\wedge e^c)$ are rewritten via (2.7), (2.10), (2.11) as
  $vol_4\wedge *e^c\big[[\phi_i,F_{bc}]^{\tilde I} e^{ib} - F^{\tilde I}_{ab} e^i_a f^c_{bi} + F^{\tilde I}_{cb} e^i_a f^a_{bi} + \tfrac12 F^{\tilde I}_{ab} f^c_{ab}\big] = vol_4\wedge *e^c\big[e^i_b\big([\phi_i,F_{bc}] - (F^{\tilde I}_{ba} f^c_{ai} + F^{\tilde I}_{ca} f^b_{ai})\big) + \tfrac12 F^{\tilde I}_{ab} f^c_{ab}\big]$ (unnumbered, PDF p.11); the last term feeds (3.28), the rest cancels by the identity (3.32) $[\phi_i,F_{ab}]^{\tilde I} = f^c_{ia}F^{\tilde I}_{cb} + f^c_{ib}F^{\tilde I}_{ac}$, which follows from $[\phi_i,\phi_a]^{\tilde I} = f_{ia}{}^c\phi^{\tilde I}_c$, the definition $F^{\tilde I}_{ab} = f^c_{ab}\phi^{\tilde I}_c + f^i_{ab}\phi^{\tilde I}_i - [\phi_a,\phi_b]^{\tilde I}$, Jacobi, complete antisymmetry of $f_{ABC}$, and reductivity.
- Bianchi (Sec. 3.5, PDF p.11): $\hat D\hat F^{\tilde I} = \hat d\hat F^{\tilde I} + f^{\tilde I}_{\tilde J\tilde K}\hat A^{\tilde J}\wedge\hat F^{\tilde K} = 0$ (3.33) reduces to (3.34) $dF^{\tilde I} + f^{\tilde I}_{\tilde J\tilde K}A^{\tilde J}\wedge F^{\tilde K} = 0$, (3.35) $dD\phi^{\tilde I}_A\wedge e^A + f^{\tilde I}_{\tilde J\tilde K}A^{\tilde J}\wedge D\phi^{\tilde K}_A\wedge e^A - f^{\tilde I}_{\tilde J\tilde K}\phi^{\tilde J}_A F^{\tilde K}\wedge e^A = 0$, (3.36) $DF^{\tilde I}_{BC} = f^A_{BC}D\phi^{\tilde I}_A - 2 f^{\tilde I}_{\tilde J\tilde K}\phi^{\tilde J}_B D\phi^{\tilde K}_C$, (3.37) $\tfrac12(F^{\tilde I}_{AB}f^A_{DC} - f^{\tilde I}_{\tilde J\tilde K}\phi^{\tilde J}_B F^{\tilde K}_{DC})e^D\wedge e^C\wedge e^B = 0$; with (3.18) and Jacobi these give $DF = 0$ and $D^2\phi = F\wedge\phi$.
- Group manifolds vs cosets (PDF p.3, p.5-6): Scherk-Schwarz reduction on a unimodular group manifold with left- (or right-) invariant fields is the known consistent case [18]; "there is no known recipe for obtaining consistent coset reductions neither a systematic understanding of the conditions under which such a reduction occurs". The CSDR ansatz (3.9) "formally looks exactly the same as the reduction ansatz on group manifolds" (PDF p.8) and the gauge-covariant version (2.16) of the invariance condition is what makes it work on $S/R$. Result applies to any reductive coset with $S$ unimodular and totally antisymmetric $f_{ABC}$ (compact semisimple $S$); gravity is held fixed.

## (d) Explicit examples

- None. No $S^2 = SU(2)/U(1)$ or other specific coset is worked out; the analysis is fully general.

## (e) Kaluza-Klein gravity formulas

- E.o.m. of (3.1), PDF p.7: (3.2) $\hat R_{MN} = \tfrac12\mathrm{Tr}\big(\hat F^2_{MN} - \tfrac18\hat F^2\hat g_{MN}\big) + \tfrac14\lambda_{(D)}\hat g_{MN}$, (3.3) $\hat D(*_D\hat F_{(2)}) = 0$.
- Background (3.4) $g_{(D)} = \eta_{mn}e^m e^n + \gamma_{ab}e^a e^b$; Einstein equations on it reduce to (3.5) $\tfrac14\mathrm{Tr}\,F_{ab}F^{ab} = \lambda_{(D)}$ and (3.6) $\mathrm{Tr}\,F^2_{ab} = R_{(d)}\gamma_{ab}$, $R_{(d)}$ the internal scalar curvature. The Ricci tensor of (3.4) is not displayed. Statement (PDF p.7): "the complete elimination of the Kaluza-Klein gauge fields as well as of the scalar moduli is a well-known consistent truncation of gravity on coset spaces [20]"; dynamical gravity on cosets deferred to [24].

## (f) Equation-number index (PDF page / printed page)

- (2.1)-(2.3): PDF p.4 / printed 3. (2.4)-(2.12): PDF p.5 / printed 4. (2.13): PDF p.5 / printed 4.
- (2.14)-(2.18): PDF p.6 / printed 5. (3.1): PDF p.6 / printed 5. Unnumbered $\hat F_{(2)}$ definition, (3.2)-(3.7): PDF p.7 / printed 6.
- (3.8)-(3.16): PDF p.8 / printed 7. (3.17)-(3.23): PDF p.9 / printed 8.
- (3.24)-(3.30): PDF p.10 / printed 9. (3.31), unnumbered two-line rewriting, (3.32)-(3.37): PDF p.11 / printed 10.
- Commutative diagram: PDF p.3 / printed 2 (unnumbered). Definition of consistent truncation: PDF p.4 / printed 3 (unnumbered, italic).
- There is no equation numbered (3.38) or higher, and no numbered equations in Sec. 1 or Sec. 4.

# Attributable statements

1. Consistency of a truncation = commutativity of "reduce then vary" with "vary then reduce"; diagram and definition (Sec. 1, PDF p.3-4).
2. Two types of truncation: KK (keep massless modes) vs constraint-driven; CSDR is of the second type (Sec. 1, PDF p.3-4).
3. Secondary constraints are the typical obstruction to consistency of constraint-driven truncations (Dirac-Bergmann analogy, after Pons-Talavera) (Sec. 1, PDF p.4).
4. Scherk-Schwarz on group manifolds keeps $G_L$ singlets, $\mathcal L_{X^I}A = 0$; CSDR relaxes to $\mathcal L_{X^I}A = DW_I$ with compatibility $[\mathcal L_{X^I},\mathcal L_{X^J}] = \mathcal L_{[X^I,X^J]}$ (Sec. 2.2, PDF p.5-6, eqs. (2.14)-(2.18)).
5. Expanding $\hat A = A + \phi_A e^A$ over all of $\mathrm{Lie}(S)$ (including the $R$-connection $e^i$) yields constant structure constants in $\hat F$ and gauge-invariant constraints; expanding only over $e^a$ would give $y$-dependent $C^a_{bc}(y)$ (Sec. 3.2, PDF p.8).
6. The reduced Lagrangian (3.17) requires exactly the constraints $D\phi_i = F_{ai} = F_{ij} = 0$, i.e. (3.19)-(3.21) (Sec. 3.2, PDF p.9).
7. 4D gauge group $H = C_G(R)$; scalars from matching $R$-irreps in $\mathrm{adj}S$ and $\mathrm{adj}G$ (Sec. 3.3, PDF p.9-10, (3.22)-(3.23)).
8. Reducing the YM e.o.m. with the same constraints gives (3.27)-(3.28), the e.o.m. of (3.17), with no secondary constraints: CSDR of Yang-Mills on a reductive coset of a unimodular $S$ is a consistent truncation (Sec. 3.4, PDF p.10-11; Sec. 4).
9. Bianchi identities reduce to $DF = 0$ and $D^2\phi = F\wedge\phi$ (Sec. 3.5, PDF p.11).
10. Removing all KK gauge fields and moduli is a known consistent truncation of gravity on cosets; including gravity fluctuations is left open (Sec. 3.1, PDF p.7; Sec. 4, PDF p.12).

# Caveats

- Gravity is frozen: the metric (3.4) is a fixed background and (3.5)-(3.6) are only consistency conditions on $\lambda_{(D)}$ and the internal curvature. The result is about the Yang-Mills sector only; the paper itself says gravity fluctuations are "substantially more involved" and in progress [24].
- The consistency proof needs: $S$ unimodular ($f^A_{AB}=0$), $f^a_{ab}=0$, $f_{ABC}$ totally antisymmetric, and reductive $S/R$; $R$ must be embeddable in $G$. These are stated in passing (PDF p.10-11), not as a theorem.
- The paper does not derive the constraints (3.18) from the symmetry condition (2.17); it refers to [13], [14]. It also skips the algebra from (3.16) to (3.17).
- (3.17) as printed has no $\mathrm{Tr}$ on the potential term $-\tfrac14 F_{ab}F^{ab}vol_4$ and uses lower-lower-lower $f_{bca}$ in (3.28); treat these as notational looseness (footnote 4: Lie-algebra indices are occasionally suppressed). Coset indices are raised/lowered with $\delta_{ab}$ (PDF p.5), so $\gamma_{ab}$ vs $\delta_{ab}$ is not carefully distinguished; no radii or rescalings appear.
- The sign conventions of $\mathrm{Tr}$ and the generator (anti-)Hermiticity are never stated; $\hat F = d\hat A + \hat A\wedge\hat A$ with structure-constant form $\tfrac12 f A\wedge A$ suggests anti-Hermitian generators with $[T_J,T_K] = f^I_{JK}T_I$ and no $i$ or $g$.
- Index letters are overloaded: $A,B,C$ are Lie-algebra indices of $S$ in Sec. 2-3 but footnote 3 calls them curved $S$ indices; $I$ labels Killing vectors in Sec. 2 but $\tilde I$ labels $G$ in Sec. 3; $F^I_{ib}$ in (3.30) has untilded $I$.
- "Consistency" here is classical and at the level of equations of motion; it says nothing about the spectrum of KK modes or stability (e.g. tachyons in the potential).
- The arXiv abstract lists three authors in the brief, but the PDF has four (Zoupanos included); cite accordingly.
