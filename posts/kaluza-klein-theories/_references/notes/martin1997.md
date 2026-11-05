# Identity

- arXiv: hep-ph/9709356 (version 7, January 2016, as printed on the title page; arXiv stamp "hep-ph/9709356v7 27 Jan 2016").
- Title: A Supersymmetry Primer.
- Author: Stephen P. Martin (Department of Physics, Northern Illinois University, DeKalb IL).
- Year: 1997 (first version); the PDF read here is the 2016 revision. 160 pages. PDF page numbers coincide with printed page numbers throughout.

# Structure

1. Introduction, pp. 3-12 (hierarchy problem pp. 3-5; superalgebra in schematic form and boson/fermion counting pp. 5-7; MSSM field content pp. 7-10; soft breaking motivation pp. 10-12).
2. Interlude: Notations and Conventions, pp. 13-17.
3. Supersymmetric Lagrangians, pp. 17-30.
   - 3.1 The simplest supersymmetric model: a free chiral supermultiplet, pp. 18-21.
   - 3.2 Interactions of chiral supermultiplets, pp. 22-25.
   - 3.3 Lagrangians for gauge supermultiplets, pp. 25-26.
   - 3.4 Supersymmetric gauge interactions, pp. 26-28.
   - 3.5 Summary: How to build a supersymmetric model, pp. 28-30.
4. Superspace and superfields, pp. 30-53 (4.1 supercoordinates p. 31; 4.2 transformations p. 33; 4.3 chiral covariant derivatives p. 35; 4.4 chiral superfields p. 37; 4.5 vector superfields p. 38; 4.6 Lagrangians in superspace p. 40; 4.7 chiral p. 41; 4.8 Abelian gauge p. 43; 4.9 general gauge p. 46; 4.10 non-renormalizable p. 49; 4.11 R symmetries p. 51).
5. Soft supersymmetry breaking interactions, pp. 53-54.
6. The Minimal Supersymmetric Standard Model, pp. 54-72 (6.1 superpotential p. 54; 6.2 R-parity p. 58; 6.3 soft breaking in the MSSM p. 60; 6.4 organizing principle p. 61; 6.5 RG equations p. 66).
7. Origins of supersymmetry breaking, pp. 72-95 (7.1 general p. 72; 7.2 Fayet-Iliopoulos p. 74; 7.3 O'Raifeartaigh p. 75; 7.4 separate sector p. 78; 7.5 goldstino and gravitino p. 80; 7.6 Planck-mediated p. 82; 7.7 gauge-mediated p. 88; 7.8 extra-dimensional and anomaly-mediated p. 92).
8. The mass spectrum of the MSSM, pp. 95-114.
9. Sparticle decays, pp. 114-119.
10. Experimental signals for supersymmetry, pp. 119-133.
11. Beyond minimal supersymmetry, pp. 133-144.
12. Concluding remarks, p. 144; Acknowledgments p. 145; References p. 145.

# Conventions

- Metric (eq. 2.3, p. 13): $\eta_{\mu\nu} = \mathrm{diag}(-1,+1,+1,+1)$, so $p^2 = -m^2$ on shell. $x^\mu = (t,\vec x)$, $p^\mu = (E,\vec p)$, $\partial_\mu = (\partial/\partial t, \vec\nabla)$ (eqs. 2.1-2.2).
- Gamma matrices in 2x2 blocks (eq. 2.6): $\gamma^\mu = \begin{pmatrix} 0 & \sigma^\mu \\ \bar\sigma^\mu & 0\end{pmatrix}$, $\gamma_5 = \mathrm{diag}(-1, 1)$.
- Sigma matrices (eq. 2.7, p. 14): $\sigma^0 = \bar\sigma^0 = \mathbf 1$, $\sigma^i = -\bar\sigma^i = $ Pauli matrices. Index heights: $(\sigma^\mu)_{\alpha\dot\alpha}$ and $(\bar\sigma^\mu)^{\dot\alpha\alpha}$.
- Dirac spinor $\Psi_D = (\xi_\alpha,\ \chi^{\dagger\dot\alpha})^T$ (eq. 2.8); $\bar\Psi_D = (\chi^\alpha,\ \xi^\dagger_{\dot\alpha})$ (eq. 2.9). Undotted indices = left-handed, dotted = right-handed. $\psi^\dagger_{\dot\alpha} \equiv (\psi_\alpha)^\dagger$ (eq. 2.11). Martin uses a dagger, not a bar, for the conjugate Weyl spinor (footnote p. 14).
- Epsilon (eq. 2.13): $\epsilon^{12} = -\epsilon^{21} = \epsilon_{21} = -\epsilon_{12} = 1$, diagonal entries zero. Raising/lowering (eq. 2.14): $\xi_\alpha = \epsilon_{\alpha\beta}\xi^\beta$, $\xi^\alpha = \epsilon^{\alpha\beta}\xi_\beta$, $\chi^\dagger_{\dot\alpha} = \epsilon_{\dot\alpha\dot\beta}\chi^{\dagger\dot\beta}$, $\chi^{\dagger\dot\alpha} = \epsilon^{\dot\alpha\dot\beta}\chi^\dagger_{\dot\beta}$; consistent because $\epsilon_{\alpha\beta}\epsilon^{\beta\gamma} = \delta_\alpha^\gamma$.
- Suppressed contractions (eq. 2.15): undotted ${}^\alpha{}_\alpha$ (down-right), dotted ${}_{\dot\alpha}{}^{\dot\alpha}$ (up-right). Hence $\xi\chi \equiv \xi^\alpha\chi_\alpha = \chi\xi$ (eq. 2.16), $(\xi\chi)^* = \chi^\dagger\xi^\dagger = \xi^\dagger\chi^\dagger$ (eq. 2.17).
- Bilinear conjugation (p. 15): $(\chi^\dagger\bar\sigma^\mu\xi)^* = \xi^\dagger\bar\sigma^\mu\chi = -\chi\sigma^\mu\xi^\dagger = -(\xi\sigma^\mu\chi^\dagger)^*$ (eq. 2.18), $(\chi^\dagger\bar\sigma^\nu\sigma^\mu\xi^\dagger)^* = \xi\sigma^\mu\bar\sigma^\nu\chi = \chi\sigma^\nu\bar\sigma^\mu\xi$ (eq. 2.19).
- Fierz identity (eq. 2.20): $\chi_\alpha(\xi\eta) = -\xi_\alpha(\eta\chi) - \eta_\alpha(\chi\xi)$.
- Reduction identities (eqs. 2.21-2.27, p. 15), e.g. $\sigma^\mu_{\alpha\dot\alpha}\bar\sigma_\mu^{\dot\beta\beta} = -2\delta_\alpha^\beta\delta_{\dot\alpha}^{\dot\beta}$ (2.21), $[\sigma^\mu\bar\sigma^\nu + \sigma^\nu\bar\sigma^\mu]_\alpha{}^\beta = -2\eta^{\mu\nu}\delta_\alpha^\beta$ (2.24), $[\bar\sigma^\mu\sigma^\nu + \bar\sigma^\nu\sigma^\mu]^{\dot\beta}{}_{\dot\alpha} = -2\eta^{\mu\nu}\delta^{\dot\beta}_{\dot\alpha}$ (2.25); $\epsilon^{0123} = +1$.
- Weyl Lagrangian (eq. 2.28): $\mathcal L_{\rm Dirac} = i\xi^\dagger\bar\sigma^\mu\partial_\mu\xi + i\chi^\dagger\bar\sigma^\mu\partial_\mu\chi - M(\xi\chi + \xi^\dagger\chi^\dagger)$; Majorana (eq. 2.31): $i\xi^\dagger\bar\sigma^\mu\partial_\mu\xi - \tfrac12 M(\xi\xi + \xi^\dagger\xi^\dagger)$, with $\xi\xi = 2\xi_2\xi_1 \neq 0$. Gauge covariant derivative $\nabla_\mu\psi_i = \partial_\mu\psi_i - ig_a A^a_\mu T^{a}{}_i{}^j\psi_j$ (eq. 2.34), $T^a$ Hermitian.
- Grassmann parameter (p. 18, below eq. 3.1.3): $\epsilon^\alpha$ is an infinitesimal, anticommuting, two-component Weyl object; global SUSY means $\partial_\mu\epsilon^\alpha = 0$ (until section 7.5); dimension $[\epsilon] = [\mathrm{mass}]^{-1/2}$ since $[\psi] = 3/2$, $[\phi] = 1$. Fields $\phi_i,\psi_i$ carry lowered flavour indices, conjugates raised (p. 22).
- Superalgebra (eqs. 3.1.30-3.1.32, p. 21): $\{Q_\alpha, Q^\dagger_{\dot\alpha}\} = -2\sigma^\mu_{\alpha\dot\alpha}P_\mu$, others zero; charges normalised as $Q_\alpha = \sqrt2\int d^3\vec x\, J^0_\alpha$ (eq. 3.1.20) so that $[\epsilon Q + \epsilon^\dagger Q^\dagger, X] = -i\sqrt2\,\delta X$ (eq. 3.1.21); the $\sqrt2$ is "an arbitrary historical convention" (p. 21). Translations: $[P^\mu, X] = i\partial^\mu X$ (eq. 3.1.27).

# Equations the tutorial uses

(a) Superalgebra. Schematic form in the Introduction (eqs. 1.6-1.8, p. 6): $\{Q,Q^\dagger\} = P^\mu$, $\{Q,Q\} = \{Q^\dagger,Q^\dagger\} = 0$, $[P^\mu,Q] = [P^\mu,Q^\dagger] = 0$. Precise form (sec. 3.1, p. 21):
$$\{Q_\alpha, Q^\dagger_{\dot\alpha}\} = -2\sigma^\mu_{\alpha\dot\alpha}P_\mu, \qquad (3.1.30)$$
$$\{Q_\alpha,Q_\beta\} = 0, \qquad \{Q^\dagger_{\dot\alpha},Q^\dagger_{\dot\beta}\} = 0, \qquad (3.1.31)$$
$$[Q_\alpha, P^\mu] = 0, \qquad [Q^\dagger_{\dot\alpha}, P^\mu] = 0. \qquad (3.1.32)$$
Derived from $[\epsilon_2 Q + \epsilon_2^\dagger Q^\dagger,\ \epsilon_1 Q + \epsilon_1^\dagger Q^\dagger] = 2(\epsilon_1\sigma_\mu\epsilon_2^\dagger - \epsilon_2\sigma_\mu\epsilon_1^\dagger)P^\mu$ (eq. 3.1.29), itself from (3.1.16), (3.1.21), (3.1.27) and the Jacobi identity (eq. 3.1.28). Canonical relations used: $[\phi(\vec x),\pi(\vec y)] = i\delta^{(3)}$, $\{\psi_\alpha(\vec x),\psi^\dagger_{\dot\alpha}(\vec y)\} = (\sigma^0)_{\alpha\dot\alpha}\delta^{(3)}(\vec x-\vec y)$ (eqs. 3.1.22-3.1.23). Positivity check in these conventions: for $P^\mu = (E,\vec 0)$, $P_0 = -E$, so $\{Q_\alpha,Q^\dagger_{\dot\alpha}\} = 2E\,\delta_{\alpha\dot\alpha}$.

(b) Free Wess-Zumino model (sec. 3.1, pp. 18-20). Action $S = \int d^4x(\mathcal L_{\rm scalar} + \mathcal L_{\rm fermion})$ (3.1.1) with
$$\mathcal L_{\rm scalar} = -\partial^\mu\phi^*\partial_\mu\phi, \qquad \mathcal L_{\rm fermion} = i\psi^\dagger\bar\sigma^\mu\partial_\mu\psi. \qquad (3.1.2)$$
Transformations:
$$\delta\phi = \epsilon\psi, \qquad \delta\phi^* = \epsilon^\dagger\psi^\dagger, \qquad (3.1.3)$$
$$\delta\psi_\alpha = -i(\sigma^\mu\epsilon^\dagger)_\alpha\partial_\mu\phi, \qquad \delta\psi^\dagger_{\dot\alpha} = i(\epsilon\sigma^\mu)_{\dot\alpha}\partial_\mu\phi^* \qquad (3.1.5)$$
(the form of $\delta\psi$ is forced: linear in $\epsilon^\dagger$ and $\phi$ with one derivative, p. 18). Invariance computation:
$$\delta\mathcal L_{\rm scalar} = -\epsilon\partial^\mu\psi\,\partial_\mu\phi^* - \epsilon^\dagger\partial^\mu\psi^\dagger\,\partial_\mu\phi, \qquad (3.1.4)$$
$$\delta\mathcal L_{\rm fermion} = -\epsilon\sigma^\mu\bar\sigma^\nu\partial_\nu\psi\,\partial_\mu\phi^* + \psi^\dagger\bar\sigma^\nu\sigma^\mu\epsilon^\dagger\,\partial_\mu\partial_\nu\phi, \qquad (3.1.6)$$
which by (2.24), (2.25) and $\partial_\mu\partial_\nu = \partial_\nu\partial_\mu$ becomes
$$\delta\mathcal L_{\rm fermion} = \epsilon\partial^\mu\psi\,\partial_\mu\phi^* + \epsilon^\dagger\partial^\mu\psi^\dagger\,\partial_\mu\phi - \partial_\mu\big(\epsilon\sigma^\nu\bar\sigma^\mu\psi\,\partial_\nu\phi^* + \epsilon\psi\,\partial^\mu\phi^* + \epsilon^\dagger\psi^\dagger\,\partial^\mu\phi\big). \qquad (3.1.7)$$
First two terms cancel (3.1.4); the rest is a total derivative, so $\delta S = 0$ (3.1.8). Closure on $\phi$:
$$(\delta_{\epsilon_2}\delta_{\epsilon_1} - \delta_{\epsilon_1}\delta_{\epsilon_2})\phi = i(-\epsilon_1\sigma^\mu\epsilon_2^\dagger + \epsilon_2\sigma^\mu\epsilon_1^\dagger)\partial_\mu\phi. \qquad (3.1.9)$$
On $\psi$ (3.1.10-3.1.11, via Fierz 2.20 and 2.18):
$$(\delta_{\epsilon_2}\delta_{\epsilon_1} - \delta_{\epsilon_1}\delta_{\epsilon_2})\psi_\alpha = i(-\epsilon_1\sigma^\mu\epsilon_2^\dagger + \epsilon_2\sigma^\mu\epsilon_1^\dagger)\partial_\mu\psi_\alpha + i\epsilon_{1\alpha}\,\epsilon_2^\dagger\bar\sigma^\mu\partial_\mu\psi - i\epsilon_{2\alpha}\,\epsilon_1^\dagger\bar\sigma^\mu\partial_\mu\psi, \qquad (3.1.11)$$
the last two terms vanishing only on shell ($\bar\sigma^\mu\partial_\mu\psi = 0$). Fix: auxiliary complex scalar $F$ with $\mathcal L_{\rm auxiliary} = F^*F$ (3.1.12), $[F] = [\mathrm{mass}]^2$, EOM $F = F^* = 0$;
$$\delta F = -i\epsilon^\dagger\bar\sigma^\mu\partial_\mu\psi, \qquad \delta F^* = i\partial_\mu\psi^\dagger\bar\sigma^\mu\epsilon; \qquad (3.1.13)$$
$\delta\mathcal L_{\rm auxiliary} = -i\epsilon^\dagger\bar\sigma^\mu\partial_\mu\psi\,F^* + i\partial_\mu\psi^\dagger\bar\sigma^\mu\epsilon\,F$ (3.1.14); modified
$$\delta\psi_\alpha = -i(\sigma^\mu\epsilon^\dagger)_\alpha\partial_\mu\phi + \epsilon_\alpha F, \qquad \delta\psi^\dagger_{\dot\alpha} = i(\epsilon\sigma^\mu)_{\dot\alpha}\partial_\mu\phi^* + \epsilon^\dagger_{\dot\alpha}F^*. \qquad (3.1.15)$$
Then for every $X = \phi,\phi^*,\psi,\psi^\dagger,F,F^*$, without equations of motion:
$$(\delta_{\epsilon_2}\delta_{\epsilon_1} - \delta_{\epsilon_1}\delta_{\epsilon_2})X = i(-\epsilon_1\sigma^\mu\epsilon_2^\dagger + \epsilon_2\sigma^\mu\epsilon_1^\dagger)\partial_\mu X. \qquad (3.1.16)$$
Table 3.1 (p. 20): on-shell $(\phi,\psi,F) = (2,2,0)$ real d.o.f., $n_B = n_F = 2$; off-shell $(2,4,2)$, $n_B = n_F = 4$. Supercurrent (3.1.17-3.1.19): $\epsilon J^\mu + \epsilon^\dagger J^{\dagger\mu} \equiv \sum_X \delta X\,\partial\mathcal L/\partial(\partial_\mu X) - K^\mu$ with $\delta\mathcal L = \partial_\mu K^\mu$; $J^\mu_\alpha = (\sigma^\nu\bar\sigma^\mu\psi)_\alpha\partial_\nu\phi^*$, $J^{\dagger\mu}_{\dot\alpha} = (\psi^\dagger\bar\sigma^\mu\sigma^\nu)_{\dot\alpha}\partial_\nu\phi$ (3.1.18).

(c) Superpotential (sec. 3.2, pp. 22-25). $\mathcal L_{\rm free} = -\partial^\mu\phi^{*i}\partial_\mu\phi_i + i\psi^{\dagger i}\bar\sigma^\mu\partial_\mu\psi_i + F^{*i}F_i$ (3.2.1), invariant under (3.2.2-3.2.4) (the $i$-indexed versions of 3.1.3, 3.1.15, 3.1.13). Most general renormalisable candidate $\mathcal L_{\rm int} = (-\tfrac12 W^{ij}\psi_i\psi_j + W^iF_i + x^{ij}F_iF_j) + \text{c.c.} - U$ (3.2.5); invariance kills $U$ and $x^{ij}$ (3.2.6). Four-spinor variation (3.2.7) plus Fierz $(\epsilon\psi_i)(\psi_j\psi_k) + (\epsilon\psi_j)(\psi_k\psi_i) + (\epsilon\psi_k)(\psi_i\psi_j) = 0$ (3.2.8) force $\delta W^{ij}/\delta\phi_k$ totally symmetric and $W^{ij}$ holomorphic (no $\phi^{*k}$). Hence $W^{ij} = M^{ij} + y^{ijk}\phi_k$ (3.2.9), $W^{ij} = \delta^2 W/\delta\phi_i\delta\phi_j$ (3.2.10) with
$$W = \tfrac12 M^{ij}\phi_i\phi_j + \tfrac16 y^{ijk}\phi_i\phi_j\phi_k \qquad (3.2.11)$$
(general with linear term: $W = L^i\phi_i + \ldots$, 3.2.15). Derivative terms $\delta\mathcal L_{\rm int}|_\partial = (iW^{ij}\partial_\mu\phi_j\,\psi_i\sigma^\mu\epsilon^\dagger + iW^i\,\partial_\mu\psi_i\sigma^\mu\epsilon^\dagger) + \text{c.c.}$ (3.2.12) are a total derivative iff $W^i = \delta W/\delta\phi_i = M^{ij}\phi_j + \tfrac12 y^{ijk}\phi_j\phi_k$ (3.2.14). Auxiliary EOM (from the $F$-dependent part $F_iF^{*i} + W^iF_i + W^*_iF^{*i}$):
$$F_i = -W^*_i, \qquad F^{*i} = -W^i. \qquad (3.2.16)$$
Eliminating:
$$\mathcal L = -\partial^\mu\phi^{*i}\partial_\mu\phi_i + i\psi^{\dagger i}\bar\sigma^\mu\partial_\mu\psi_i - \tfrac12(W^{ij}\psi_i\psi_j + W^*_{ij}\psi^{\dagger i}\psi^{\dagger j}) - W^iW^*_i, \qquad (3.2.17)$$
$$V(\phi,\phi^*) = W^kW^*_k = F^{*k}F_k = M^*_{ik}M^{kj}\phi^{*i}\phi_j + \tfrac12 M^{in}y^*_{jkn}\phi_i\phi^{*j}\phi^{*k} + \tfrac12 M^*_{in}y^{jkn}\phi^{*i}\phi_j\phi_k + \tfrac14 y^{ijn}y^*_{kln}\phi_i\phi_j\phi^{*k}\phi^{*l}, \qquad (3.2.18)$$
Full Lagrangian after substituting (3.2.11):
$$\mathcal L = -\partial^\mu\phi^{*i}\partial_\mu\phi_i - V(\phi,\phi^*) + i\psi^{\dagger i}\bar\sigma^\mu\partial_\mu\psi_i - \tfrac12 M^{ij}\psi_i\psi_j - \tfrac12 M^*_{ij}\psi^{\dagger i}\psi^{\dagger j} - \tfrac12 y^{ijk}\phi_i\psi_j\psi_k - \tfrac12 y^*_{ijk}\phi^{*i}\psi^{\dagger j}\psi^{\dagger k}. \qquad (3.2.19)$$ Linearised EOM (3.2.20-3.2.22): $\partial^\mu\partial_\mu\phi_i = M^*_{ik}M^{kj}\phi_j + \ldots$, $\partial^\mu\partial_\mu\psi_i = M^*_{ik}M^{kj}\psi_j + \ldots$, same squared-mass matrix $(M^2)_i{}^j = M^*_{ik}M^{kj}$ (p. 25).

(d) Gauge supermultiplet (sec. 3.3-3.4, pp. 25-28).
$$\mathcal L_{\rm gauge} = -\tfrac14 F^a_{\mu\nu}F^{\mu\nu a} + i\lambda^{\dagger a}\bar\sigma^\mu\nabla_\mu\lambda^a + \tfrac12 D^aD^a, \qquad (3.3.3)$$
$F^a_{\mu\nu} = \partial_\mu A^a_\nu - \partial_\nu A^a_\mu + gf^{abc}A^b_\mu A^c_\nu$ (3.3.4), $\nabla_\mu\lambda^a = \partial_\mu\lambda^a + gf^{abc}A^b_\mu\lambda^c$ (3.3.5). Transformations (3.3.6-3.3.8):
$$\delta A^a_\mu = -\tfrac{1}{\sqrt2}(\epsilon^\dagger\bar\sigma_\mu\lambda^a + \lambda^{\dagger a}\bar\sigma_\mu\epsilon), \qquad \delta\lambda^a_\alpha = \tfrac{i}{2\sqrt2}(\sigma^\mu\bar\sigma^\nu\epsilon)_\alpha F^a_{\mu\nu} + \tfrac{1}{\sqrt2}\epsilon_\alpha D^a, \qquad \delta D^a = \tfrac{i}{\sqrt2}(-\epsilon^\dagger\bar\sigma^\mu\nabla_\mu\lambda^a + \nabla_\mu\lambda^{\dagger a}\bar\sigma^\mu\epsilon).$$
The $\sqrt2$ factors are fixed by invariance of the action; the phase of $\lambda^a$ is chosen for MSSM convenience (p. 26). Closure (3.3.9) with $\nabla_\mu X$ in place of $\partial_\mu X$. Table 3.2: on-shell $(A_\mu,\lambda,D) = (2,2,0)$, off-shell $(3,4,1)$. Coupling to matter: $\nabla_\mu\phi_i = \partial_\mu\phi_i - igA^a_\mu(T^a\phi)_i$ etc. (3.4.2-3.4.4); $\delta F_i = -i\epsilon^\dagger\bar\sigma^\mu\nabla_\mu\psi_i + \sqrt2 g(T^a\phi)_i\,\epsilon^\dagger\lambda^{\dagger a}$ (3.4.8);
$$\mathcal L = \mathcal L_{\rm chiral} + \mathcal L_{\rm gauge} - \sqrt2 g(\phi^*T^a\psi)\lambda^a - \sqrt2 g\lambda^{\dagger a}(\psi^\dagger T^a\phi) + g(\phi^*T^a\phi)D^a, \qquad (3.4.9)$$
requires $W^i(T^a\phi)_i = 0$ (3.4.10, gauge invariance of $W$). $D$ EOM: $D^a = -g(\phi^*T^a\phi)$ (3.4.11). Scalar potential:
$$V(\phi,\phi^*) = F^{*i}F_i + \tfrac12\sum_a D^aD^a = W^*_iW^i + \tfrac12\sum_a g_a^2(\phi^*T^a\phi)^2 \qquad (3.4.12)$$
("F-term" and "D-term"; always $\ge 0$; fixed entirely by the other interactions, p. 28). Supercurrent of the full theory: (3.4.13).

(e) Counting argument (pp. 6-7). $(-1)^{2s}$ has eigenvalue $+1$ on bosons, $-1$ on fermions, so anticommutes with $Q,Q^\dagger$. Within a supermultiplet subspace of fixed $p^\mu$ (completeness $\sum_i|i\rangle\langle i| = 1$):
$$\sum_i\langle i|(-1)^{2s}P^\mu|i\rangle = \sum_i\langle i|(-1)^{2s}QQ^\dagger|i\rangle + \sum_i\langle i|(-1)^{2s}Q^\dagger Q|i\rangle = \sum_i\langle i|(-1)^{2s}QQ^\dagger|i\rangle - \sum_j\langle j|(-1)^{2s}QQ^\dagger|j\rangle = 0, \qquad (1.9)$$
and since $\sum_i\langle i|(-1)^{2s}P^\mu|i\rangle = p^\mu\,\mathrm{Tr}[(-1)^{2s}] \propto n_B - n_F$, one gets $n_B = n_F$ (1.10) for $p^\mu \neq 0$. Consequences listed p. 7: chiral multiplet (Weyl fermion + complex scalar), gauge multiplet (massless vector + Weyl gaugino in the adjoint), gravity multiplet (graviton + gravitino); footnote on massive vector multiplet (3+1 bosonic, 4 fermionic).

(f) Mass degeneracy (p. 6, and sec. 3.2 pp. 24-25). $-P^2$ commutes with $Q,Q^\dagger$ and with all Poincare generators, so all members of an irreducible supermultiplet have the same $-P^2$ eigenvalue, hence equal masses; they also share gauge quantum numbers since $Q$ commutes with gauge generators. Explicit example: in the interacting Wess-Zumino model the scalars and fermions satisfy the same wave equation with squared-mass matrix $(M^2)_i{}^j = M^*_{ik}M^{kj}$ (3.2.20-3.2.22); diagonalising by a unitary redefinition gives mass-degenerate (complex scalar, Weyl fermion) pairs.

# Attributable statements

1. The hierarchy problem is "not really a difficulty with the Standard Model itself, but rather a disturbing sensitivity of the Higgs potential to new physics"; $\Delta m_H^2 = -\frac{|\lambda_f|^2}{8\pi^2}\Lambda_{\rm UV}^2 + \ldots$ for a fermion loop (eq. 1.2) and $+\frac{\lambda_S}{16\pi^2}[\Lambda_{\rm UV}^2 - 2m_S^2\ln(\Lambda_{\rm UV}/m_S)+\ldots]$ for a scalar loop (eq. 1.3); the relative sign suggests a boson-fermion symmetry (sec. 1, pp. 3-5).
2. The absence of quadratic divergences is "a necessary, but not sufficient, criterion for avoiding the hierarchy problem"; the real issue is quadratic sensitivity to heavy mass scales (sec. 1, p. 4).
3. The Haag-Lopuszanski-Sohnius extension of the Coleman-Mandula theorem forces the generators of a boson-fermion symmetry in a chiral theory to satisfy the algebra (1.6)-(1.8) (sec. 1, p. 6).
4. Particles in the same irreducible supermultiplet have equal masses and identical gauge quantum numbers (sec. 1, p. 6).
5. Each supermultiplet contains equal numbers of bosonic and fermionic degrees of freedom, $n_B = n_F$, by the $(-1)^{2s}$ trace argument (eqs. 1.9-1.10, pp. 6-7).
6. The commutator of two supersymmetry transformations is a spacetime translation: "we have found that the commutator of two supersymmetry transformations gives us back the derivative of the original field" (eq. 3.1.9, p. 19).
7. The algebra closes only on shell without $F$; the auxiliary field is "really just [a] book-keeping device" that lets it close off-shell and matches the off-shell counting $n_B = n_F = 4$ (sec. 3.1, pp. 19-20, Table 3.1).
8. The most general renormalisable non-gauge interactions of chiral multiplets are fixed by a single holomorphic function, the superpotential $W$; holomorphy follows from the Fierz identity (3.2.8) and the absence of any cancelling term for $\delta W^{ij}/\delta\phi^{*k}$ (sec. 3.2, p. 23).
9. The scalar potential $V = W^*_iW^i + \tfrac12\sum_a g_a^2(\phi^*T^a\phi)^2$ is a sum of squares, bounded below, and "completely determined by the other interactions in the theory" (eq. 3.4.12, p. 28).
10. The gaugino-matter couplings $-\sqrt2 g(\phi^*T^a\psi)\lambda^a + \text{c.c.}$ have strengths fixed by supersymmetry to be gauge couplings though they are not gauge interactions in the ordinary sense (eq. 3.4.9, p. 27).
11. Extended ($N > 1$) four-dimensional supersymmetry cannot accommodate chiral fermions, though it "might describe the real world if the extra dimensions are compactified in an appropriate way" (sec. 1, pp. 7-8). Useful bridge to compactification.

# Caveats

- Metric is mostly-plus, $\eta = \mathrm{diag}(-,+,+,+)$, with $\sigma^0 = +\mathbf 1$ and $\sigma^i = $ Pauli. This gives the minus sign in $\{Q_\alpha,Q^\dagger_{\dot\alpha}\} = -2\sigma^\mu_{\alpha\dot\alpha}P_\mu$. Wess-Bagger also use mostly-plus but take $\sigma^0 = -\mathbf 1$ and write $\{Q_\alpha,\bar Q_{\dot\alpha}\} = 2\sigma^\mu_{\alpha\dot\alpha}P_\mu$; mostly-minus texts (e.g. Bailin-Love, Terning) have $\sigma^0 = +\mathbf 1$ and $+2\sigma^\mu P_\mu$. All agree on the physical content $\{Q_\alpha,Q^\dagger_{\dot\alpha}\} = 2E\,\delta_{\alpha\dot\alpha}$ at rest; the tutorial should state its own convention and verify positivity of energy rather than copy a sign.
- Martin writes the conjugate Weyl spinor as $\psi^\dagger_{\dot\alpha}$ where Wess-Bagger and most SUSY literature write $\bar\psi_{\dot\alpha}$ (footnote p. 14). His $\epsilon$ conventions ($\epsilon^{12} = \epsilon_{21} = +1$) match Wess-Bagger; his contraction rules $\xi\chi = \xi^\alpha\chi_\alpha$, $\xi^\dagger\chi^\dagger = \xi^\dagger_{\dot\alpha}\chi^{\dagger\dot\alpha}$ also match.
- The $\sqrt2$ in $Q_\alpha = \sqrt2\int d^3x\,J^0_\alpha$ and in $[\epsilon Q + \epsilon^\dagger Q^\dagger, X] = -i\sqrt2\,\delta X$ is declared a historical convention (p. 21); it is what makes (3.1.30) come out with coefficient 2 rather than 1. The relative factor between $\delta X$ as defined in (3.1.3)/(3.1.15) and the operator statement matters when comparing with superspace $\delta = \epsilon Q + \epsilon^\dagger Q^\dagger$ in sec. 4.2.
- Closure is on-shell only without $F$: eq. (3.1.11) contains terms proportional to $\bar\sigma^\mu\partial_\mu\psi$. With $F$ and the modified $\delta\psi$ (3.1.15) closure (3.1.16) holds off-shell on all fields. The derivation of the algebra as operator relations (3.1.29)-(3.1.32) nevertheless uses Hamiltonian equations of motion ("up to terms that vanish on-shell"), while invariance of the Lagrangian is exact off-shell (p. 21).
- $F$ has no kinetic term, $[F] = [\mathrm{mass}]^2$, and is eliminated algebraically; quadratic appearance means path-integrating it out gives the same result (footnote p. 24). For the gauge multiplet the analogous real $D^a$ is needed because $A_\mu^a$ has 3 off-shell real d.o.f. versus 4 for $\lambda^a$ (Table 3.2).
- The transformation laws (3.3.6)-(3.3.8) are in Wess-Zumino gauge and are non-linear for non-Abelian groups (footnote p. 26); linear transformations need the extra auxiliary fields of the superfield formalism (secs. 4.5, 4.8, 4.9).
- Invariance of $\mathcal L_{\rm int}$ is shown in pieces: four-spinor terms (3.2.7), derivative terms (3.2.12), and terms linear in $F$ (only stated to cancel "easily", not displayed, p. 23). The total-derivative term $K^\mu$ in the Noether procedure is explicitly non-unique (p. 20).
- Linear superpotential terms $L^i\phi_i$ (3.2.15) are allowed only for gauge singlets; relevant for F-term breaking (sec. 7.1), not for the MSSM.
- No equation number exists for the statement $n_B = n_F$ beyond (1.10); the summary "boson-fermion degeneracy in the WZ model" at the top of p. 25 is prose, not a numbered equation.
