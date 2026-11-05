# Identity

- arXiv: hep-th/0510040 (v4, 20 Feb 2013).
- Title: Introductory Lectures on Quantum Field Theory.
- Authors: Luis Alvarez-Gaume (CERN) and Miguel A. Vazquez-Mozo (Universidad de Salamanca / IUFFyM).
- Year: 2005 (first version); the v4 text is based on lectures at the CERN-CLAF schools of 2005, 2009, 2011 and the 2012 Asia-Europe-Pacific School (footnote on p.1). Venue: not stated in the PDF beyond the school lectures; an expanded book version is their ref. [1], Springer 2011.
- PDF: 112 pages. PDF page n = printed page n (title page is printed p.1).

# Structure

(PDF page = printed page throughout.)

1. Introduction: p.3. 1.1 A note about notation: p.4 (eqs 1.1-1.2).
2. Why do we need Quantum Field Theory after all?: p.4-10 (eqs 2.1-2.22; Klein-Gordon equation 2.1, KG current 2.4, Klein paradox, microcausality).
3. From classical to quantum fields: p.10-16 (eqs 3.1-3.42; Fock space, invariant measure, scalar field axioms, commutator function). 3.1 Canonical quantization: p.16-19 (eqs 3.43-3.58). 3.2 The Casimir effect: p.19-21 (eqs 3.59-3.66).
4. Theories and Lagrangians: p.21. 4.1 Representations of the Lorentz group: p.21-23. 4.2 Spinors: p.23-29 (Dirac Lagrangian 4.27 on p.26; conserved current 4.45-4.46 on p.29). 4.3 Gauge fields: p.29-36 (Maxwell 4.47-4.51; minimal coupling 4.52-4.53; Aharonov-Bohm 4.54-4.55; Dirac monopole 4.56-4.61; Maxwell Lagrangian and quantisation 4.62-4.67; coupling to matter 4.68-4.72; non-abelian gauge theories 4.73-4.79). 4.4 Understanding gauge symmetry: p.36-43 (constrained Hamiltonian systems 4.80-4.86; Maxwell as a constrained system 4.87-4.96; topology of the gauge group, theta vacua 4.97-4.119).
5. Towards computational rules: Feynman diagrams: p.43. 5.1 Cross sections and S-matrix amplitudes: p.43-46. 5.2 Feynman rules: p.46-52. 5.3 An example: Compton scattering: p.52-57.
6. Symmetries: p.57. 6.1 Noether's theorem: p.57-60 (eqs 6.1-6.20). 6.2 Symmetries in the quantum theory: p.60-66 (Wigner-Weyl 6.21-6.25; Nambu-Goldstone and Goldstone's theorem 6.26-6.42; the Higgs mechanism, Abelian Higgs model 6.43-6.49, p.64-66).
7. Anomalies: p.66-67 (intro). 7.1 Axial anomaly: p.67-73 (eqs 7.1-7.x). 7.2 Chiral symmetry in QCD: p.73-80 (ends with 7.60).
8. Renormalization: p.80. 8.1 Removing infinities: p.80-84 (eqs 8.1-8.13). 8.2 The beta-function and asymptotic freedom: p.84-88 (eqs 8.14-8.17 ff.). 8.3 The renormalization group: p.88-96.
9. Special topics: p.96. 9.1 Creation of particles by classical fields: p.96-102. 9.2 Supersymmetry: p.102-106 (ends with 9.42).
- Appendix: A crash course in Group Theory: p.106-110 (eqs A.43-A.64; generators A.46-A.47, SU(2) and SU(3) examples A.48-A.54, Casimir and index A.57-A.61, anomaly coefficient A.62-A.64).
- References [1]-[49]: p.110-112.

# Conventions

- Signature: $\eta_{\mu\nu} = \mathrm{diag}(1,-1,-1,-1)$ (sec. 1.1, p.4). $\partial_\mu \equiv \partial/\partial x^\mu = (\tfrac{1}{c}\partial_t, \vec\nabla)$ (1.1). Greek indices $0,\dots,3$, Latin spatial. $\sigma^\mu = (\mathbf 1, \sigma^i)$ with the Pauli matrices (1.2). Slash notation $\not a = \gamma^\mu a_\mu$.
- Units: natural units $\hbar = c = 1$ unless stated otherwise (p.4).
- Generators: Hermitian, $(T^a)^\dagger = T^a$, group elements $U = \exp(i\alpha_a T^a)$ (6.18, p.59); $U = e^{i\chi^a(x)T^a}$ for gauge transformations (4.74, p.35). Lie algebra $[T^a, T^b] = i f^{abc} T^c$ (4.73, p.35; A.47, p.107). The Appendix writes $D(g) = \exp(-i\alpha_a T^a)$ (A.46, p.107), opposite sign of the exponent to (6.18); the two are related by $\alpha \to -\alpha$.
- Normalisation: $\mathrm{Tr}\, T^a_R T^b_R = T_2(R)\,\delta^{ab}$ (A.58, p.109), $\sum_a T^a_R T^a_R = C_2(R)\mathbf 1$ (A.57), $C_2(R)\dim R = T_2(R)\dim G$ (A.59). For SU(N): $C_2(\mathrm{fund}) = \frac{N^2-1}{2N}$, $T_2(\mathrm{fund}) = \frac12$ (A.60); $C_2(\mathrm{adj}) = T_2(\mathrm{adj}) = N$ (A.61). So the fundamental generators are normalised as $\mathrm{Tr}(T^aT^b) = \tfrac12\delta^{ab}$, e.g. $T^a(\mathbf 3) = \tfrac12\lambda_a$ for SU(3) (A.54, p.108), $\tfrac12\sigma^i$ for SU(2) (A.48).
- Abelian covariant derivative: $D_\mu = \partial_\mu + i e A_\mu$ (4.69, p.34), acting on $\psi$ with $\psi \to e^{-ie\varepsilon(x)}\psi$, $A_\mu \to A_\mu + \partial_\mu\varepsilon(x)$ (4.71, p.35). Charge $e$ sits in $D_\mu$, not in the gauge transformation of $A_\mu$.
- Non-abelian covariant derivative: $D_\mu\Phi = \partial_\mu\Phi - i g A^a_\mu T^a \Phi$ (4.78, p.36), with $A_\mu \equiv A^a_\mu T^a$ and coupling $g$. Note the opposite sign relative to the abelian (4.69): with $e = g$ the two conventions differ by $A_\mu \to -A_\mu$ (see Caveats).
- Field strength: $F^a_{\mu\nu} = \partial_\mu A^a_\nu - \partial_\nu A^a_\mu + g f^{abc} A^b_\mu A^c_\nu$ (4.75, p.35); abelian $F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu$ (p.29).
- Lagrangian normalisation: $\mathcal L_{\rm Maxwell} = -\tfrac14 F_{\mu\nu}F^{\mu\nu}$ (4.62, p.33); Yang-Mills $-\tfrac14 F^a_{\mu\nu}F^{\mu\nu\,a}$ (4.79, p.36; 4.117, p.42), i.e. the sum over $a$ is written out explicitly rather than as a trace.
- Dirac: $\gamma^\mu$ in the chiral (Weyl) basis (4.25, p.26), $\{\gamma^\mu,\gamma^\nu\} = 2\eta^{\mu\nu}$ (4.29), $\gamma^5 = -i\gamma^0\gamma^1\gamma^2\gamma^3 = \mathrm{diag}(\mathbf 1,-\mathbf 1)$ (4.30), $\overline\psi = \psi^\dagger\gamma^0$ (4.26). $\mathcal L_{\rm Dirac} = \overline\psi(i\gamma^\mu\partial_\mu - m)\psi$ (4.27, p.26).

# Equations the tutorial uses

## (a) Scalar action and Euler-Lagrange equations (sec. 3.1, p.16-17)

Free real scalar of mass $m$:
$$S[\phi(x)] \equiv \int d^4x\,\mathcal L(\phi,\partial_\mu\phi) = \frac12\int d^4x\,\left(\partial_\mu\phi\,\partial^\mu\phi - m^2\phi^2\right). \tag{3.43}$$
$$\partial_\mu\left[\frac{\partial\mathcal L}{\partial(\partial_\mu\phi)}\right] - \frac{\partial\mathcal L}{\partial\phi} = 0 \quad\Longrightarrow\quad (\partial_\mu\partial^\mu + m^2)\phi = 0. \tag{3.44}$$
Canonical momentum $\pi(x) \equiv \partial\mathcal L/\partial(\partial_0\phi) = \partial_t\phi$ (3.45); Hamiltonian $H = \frac12\int d^3x\,[\pi^2 + (\vec\nabla\phi)^2 + m^2]$ (3.46; the printed last term reads $m^2$, evidently $m^2\phi^2$). The Klein-Gordon equation also appears as $(\partial_t^2 - \nabla^2 + m^2)\psi = 0$ (2.1, p.4), with the (indefinite) current $j_\mu = \frac{i}{2}(\psi^*\partial_\mu\psi - \partial_\mu\psi^*\,\psi)$ (2.4, p.4). Plane waves $e^{-ip\cdot x}$, $E = \pm\omega_p \equiv \pm\sqrt{\vec p^{\,2}+m^2}$ (2.3).

## (b) Noether's theorem (sec. 6.1, p.57-60)

Mechanics: invariance $\delta_\epsilon L = 0$ gives $\dot Q = 0$ with $Q \equiv \frac{\partial L}{\partial\dot q_i}\delta_\epsilon q_i$ (6.1-6.2). Field theory: $\delta_\epsilon\phi$ is a symmetry if, without using the equations of motion,
$$\delta_\epsilon\mathcal L = \partial_\mu K^\mu. \tag{6.3}$$
$$\partial_\mu K^\mu = \partial_\mu\left(\frac{\partial\mathcal L}{\partial(\partial_\mu\phi)}\delta_\epsilon\phi\right) + \left[\frac{\partial\mathcal L}{\partial\phi} - \partial_\mu\left(\frac{\partial\mathcal L}{\partial(\partial_\mu\phi)}\right)\right]\delta_\epsilon\phi. \tag{6.4}$$
$$\partial_\mu J^\mu = 0 \quad\text{with}\quad J^\mu = \frac{\partial\mathcal L}{\partial(\partial_\mu\phi)}\delta_\epsilon\phi - K^\mu. \tag{6.5}$$
$$Q \equiv \int d^3x\,J^0(t,\vec x), \qquad \frac{dQ}{dt} = \int d^3x\,\partial_0 J^0 = -\int d^3x\,\partial_i J^i = 0 \tag{6.6, 6.7}$$
(fields vanishing fast enough at infinity); $Q$ is a Lorentz scalar, and after quantisation it generates the symmetry: $\delta\phi = i[\phi, Q]$ (6.8). Translations $x'^\mu = x^\mu + a^\mu$: $\delta\phi = -a^\mu\partial_\mu\phi$ (6.9), $\delta\mathcal L = -a^\mu\partial_\mu\mathcal L$ (6.10), $J^\mu = -a_\nu T^{\mu\nu}$ with
$$T^{\mu\nu} = \frac{\partial\mathcal L}{\partial(\partial_\mu\phi)}\partial^\nu\phi - \eta^{\mu\nu}\mathcal L, \qquad \partial_\mu T^{\mu\nu} = 0, \qquad P^\nu = \int d^3x\,T^{0\nu}. \tag{6.12, 6.13, 6.14}$$
U(1) example: the paper's worked example is the Dirac field, not the complex scalar. Global phase invariance of (4.27), $\psi \to e^{i\theta}\psi$, $\delta_\theta\psi = i\theta\psi$, $\delta_\theta\overline\psi = -i\theta\overline\psi$, gives
$$j^\mu = \overline\psi\gamma^\mu\psi, \quad \partial_\mu j^\mu = 0, \qquad Q = \int d^3x\,\overline\psi\gamma^0\psi = \int d^3x\,\psi^\dagger\psi \tag{6.15, 6.16}$$
(also 4.45-4.46, p.29, where $Q = e\int d^3x\,j^0$ is the electric charge). Non-abelian global symmetry $\psi_i \to U_{ij}\psi_j$, $U = \exp(i\alpha_a T^a)$: $j^{\mu a} = \overline\psi_i T^a_{ij}\gamma^\mu\psi_j$, $\partial_\mu j^{\mu a} = 0$ (6.17-6.19). The complex-scalar U(1) current is not derived via Noether in the paper; the closest is (2.4) above.

## (c) Abelian gauge invariance (sec. 4.3, p.29-35)

Potentials $A^\mu = (\varphi, \vec A)$, $F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu$ (p.29); Maxwell's equations $\partial_\mu F^{\mu\nu} = j^\nu$, $\epsilon^{\mu\nu\sigma\eta}\partial_\nu F_{\sigma\eta} = 0$ (4.49). Gauge transformation $A_\mu \to A_\mu + \partial_\mu\epsilon$ (4.50). Minimal substitution $\vec p \to \vec p - e\vec A$ in the Schrodinger equation (4.52) with $\Psi \to e^{-ie\epsilon(t,\vec x)}\Psi$ (4.53).
$$\mathcal L_{\rm Maxwell} = -\frac14 F_{\mu\nu}F^{\mu\nu} = \frac12\left(\vec E^{\,2} - \vec B^{\,2}\right). \tag{4.62}$$
Promoting the global U(1) of the Dirac Lagrangian to a local one, $\psi \to e^{-ie\varepsilon(x)}\psi$, requires $D_\mu[e^{-ie\varepsilon(x)}\psi] = e^{-ie\varepsilon(x)}D_\mu\psi$ (4.68), solved by
$$D_\mu = \partial_\mu + i e A_\mu. \tag{4.69}$$
$$\mathcal L_{\rm QED} = -\frac14 F_{\mu\nu}F^{\mu\nu} + \overline\psi(i\not D - m)\psi, \qquad \psi \to e^{-ie\varepsilon(x)}\psi, \quad A_\mu \to A_\mu + \partial_\mu\varepsilon(x), \tag{4.70, 4.71}$$
with interaction $\mathcal L^{(\rm int)}_{\rm QED} = -eA_\mu\overline\psi\gamma^\mu\psi$ (4.72).

## (d) Non-abelian gauge theory (sec. 4.3, p.35-36)

Gauge group $G$ with generators $T^a$, $a = 1,\dots,\dim G$, $[T^a,T^b] = if^{abc}T^c$ (4.73). $A_\mu \equiv A^a_\mu T^a$ with finite transformation
$$A_\mu \longrightarrow -\frac{1}{ig}U\partial_\mu U^{-1} + U A_\mu U^{-1}, \qquad U = e^{i\chi^a(x)T^a}. \tag{4.74}$$
$$F^a_{\mu\nu} = \partial_\mu A^a_\nu - \partial_\nu A^a_\mu + g f^{abc}A^b_\mu A^c_\nu, \tag{4.75}$$
which reduces to QED for $f^{abc} = 0$; it is not gauge invariant but covariant: with $F_{\mu\nu} = F^a_{\mu\nu}T^a$,
$$F_{\mu\nu} \longrightarrow U F_{\mu\nu} U^{-1}. \tag{4.76}$$
Matter in a representation: $\Phi \to U\Phi$ (4.77), $D_\mu\Phi = \partial_\mu\Phi - igA^a_\mu T^a\Phi$ (4.78). Generic (Standard-Model-type) Lagrangian
$$\mathcal L = -\frac14 F^a_{\mu\nu}F^{\mu\nu\,a} + i\overline\psi\not D\psi + \overline{D_\mu\phi}D^\mu\phi - \overline\psi\left[M_1(\phi) + i\gamma_5 M_2(\phi)\right]\psi - V(\phi), \tag{4.79}$$
with $M_{1,2}$ at most linear and $V$ at most quartic for renormalisability. Theta term: $\mathcal L = -\frac14 F^a_{\mu\nu}F^{\mu\nu\,a} - \frac{\theta g^2}{32\pi^2}F^a_{\mu\nu}\widetilde F^{\mu\nu\,a}$, $\widetilde F^a_{\mu\nu} = \frac12\epsilon_{\mu\nu\sigma\lambda}F^{\sigma\lambda}$ (4.117-4.118, p.42-43), a total derivative with $|\theta| < 10^{-9}$ (4.119).
NOT in the paper: the infinitesimal non-abelian transformation of $A_\mu$ (only the abelian $\delta A_\mu = \partial_\mu\varepsilon$, p.39), the Bianchi identity $D_{[\mu}F_{\nu\rho]} = 0$, and $F_{\mu\nu}$ as a commutator of covariant derivatives. For the tutorial's cross-check (my derivation, not the paper's): expanding (4.74) to first order in $\chi$ gives $\delta A_\mu = \frac1g\partial_\mu\chi + i[\chi, A_\mu]$, i.e. $\delta A^a_\mu = \frac1g\partial_\mu\chi^a - f^{abc}\chi^b A^c_\mu$, and $[D_\mu, D_\nu] = -igF^a_{\mu\nu}T^a$ with (4.75) and (4.78); both are consistent with the paper's signs.

## (e) Spontaneous symmetry breaking and the Higgs mechanism (sec. 6.2, p.60-66)

Wigner-Weyl: $Q^a|0\rangle = 0$ (6.22), spectrum in multiplets of equal mass. Nambu-Goldstone: $e^{i\alpha_aQ^a}|0\rangle \ne |0\rangle \Rightarrow Q^a|0\rangle \ne 0$ (6.26). Model: $N$ real scalars, $\mathcal L = \frac12\partial_\mu\varphi^i\partial^\mu\varphi^i - V(\varphi)$ (6.27), $V$ bounded below, global SO($N$) with $\delta\varphi^i = \epsilon^a(T^a)^i{}_j\varphi^j$ (6.28). Vacua: constant fields minimising $V$, $V(\langle\varphi\rangle) = 0$, $\partial V/\partial\varphi^i|_{\langle\varphi\rangle} = 0$ (6.31). Unbroken generators $H^\alpha$: $(H^\alpha)^i{}_j\langle\varphi^j\rangle = 0$ (6.32), forming a subalgebra; broken $K^A$: $(K^A)^i{}_j\langle\varphi^j\rangle \ne 0$ (6.33). Mass matrix $M^2_{ij} \equiv \partial^2V/\partial\varphi^i\partial\varphi^j|_{\varphi=\langle\varphi\rangle}$ (6.35). Invariance of $V$ (6.36), differentiated and evaluated in the vacuum, gives
$$M^2_{ik}(T^a)^i{}_j\langle\varphi^j\rangle = 0, \qquad\text{hence}\qquad M^2_{ik}(K^A)^i{}_j\langle\varphi^j\rangle = 0. \tag{6.38, 6.39}$$
Goldstone's theorem (p.62-63): "for each generator broken by the vacuum expectation value there is a massless excitation"; the mass matrix has as many zero modes as broken generators (classical proof; quantum version uses the effective action). Example: SO(3) mexican hat
$$V(\vec\varphi) = \frac{\lambda}{4}\left(\vec\varphi^{\,2} - a^2\right)^2, \qquad \langle\vec\varphi\rangle^2 = a^2, \qquad \mathrm{SO(3)} \to \mathrm{SO(2)}, \tag{6.40, 6.41}$$
two Goldstone bosons. The tutorial's form $V = -\mu^2|\phi|^2 + \lambda|\phi|^4$ does not appear; the paper always uses the completed-square form.
Higgs mechanism (p.64-66): a term $m^2A_\mu A^\mu$ is incompatible with gauge invariance; the Glashow SU(2)xU(1) model nevertheless needs two massive vectors (qualitative only, no SU(2) Higgs Lagrangian or mass matrix is given). Abelian Higgs model:
$$\mathcal L = -\frac14 F_{\mu\nu}F^{\mu\nu} + \overline{D_\mu\Phi}D^\mu\Phi - \frac{\lambda}{4}\left(\overline\Phi\Phi - \mu^2\right)^2, \tag{6.43}$$
with $D_\mu$ from (4.69); gauge transformations $\Phi \to e^{i\alpha(x)}\Phi$, $A_\mu \to A_\mu + \partial_\mu\alpha(x)$ (6.44). Minimum $|\Phi| = \mu$; no vacuum is gauge invariant, $\langle\Phi\rangle = \mu e^{i\vartheta_0} \to \mu e^{i\vartheta_0 + i\alpha(x)}$ (6.45). Expand around $\langle\Phi\rangle = \mu$:
$$\Phi(x) = \left[\mu + \frac{1}{\sqrt2}\sigma(x)\right]e^{i\vartheta(x)}, \tag{6.46}$$
gauge away $\vartheta$ with $\alpha(x) = -\vartheta(x)$ (unitary gauge), then
$$\mathcal L = -\frac14F_{\mu\nu}F^{\mu\nu} + e^2\mu^2A_\mu A^\mu + \frac12\partial_\mu\sigma\partial^\mu\sigma - \frac12\lambda\mu^2\sigma^2 - \lambda\mu\sigma^3 - \frac{\lambda}{4}\sigma^4 + e^2\mu A_\mu A^\mu\sigma + e^2A_\mu A^\mu\sigma^2, \tag{6.47}$$
$$m_\gamma^2 = 2e^2\mu^2. \tag{6.48}$$
The Higgs mass is not written as a numbered formula; reading the $\sigma^2$ term of (6.47) gives $m_\sigma^2 = \lambda\mu^2$. Degrees of freedom: $2 + 2 \to 3 + 1$; the would-be Goldstone $\vartheta$ becomes the longitudinal polarisation ("eaten"). Finite-energy configurations tend to pure gauge $\vec A \to -\frac{1}{ig}g(\vec x)\vec\nabla g(\vec x)^{-1}$ at infinity (6.49); what is broken is $\mathcal G_\infty \equiv \mathcal G/\mathcal G_0$, while Gauss' law $\vec\nabla\cdot\vec E_a|{\rm phys}\rangle = 0$ still holds. Superconductivity: $\mu^2(T)$ changes sign at $T_c$, Meissner length $1/m_\gamma$ (p.66).

## (f) Quantisation, renormalisation, anomalies (one-paragraph level)

- Canonical quantisation (sec. 3.1, p.16-18): replace Poisson brackets by commutators, $i\{\cdot,\cdot\} \to [\cdot,\cdot]$ (3.51); equal-time brackets $\{\phi(t,\vec x),\pi(t,\vec x')\} = \delta(\vec x - \vec x')$ (3.50) become $[\phi(t,\vec x),\partial_t\phi(t,\vec y)] = i\delta(\vec x - \vec y)$ (3.38). General solution $\phi(x) = \int\frac{d^3p}{(2\pi)^3}\frac{1}{2\omega_p}[\alpha(\vec p)e^{-i\omega_pt + i\vec p\cdot\vec x} + \alpha(\vec p)^*e^{i\omega_pt - i\vec p\cdot\vec x}]$ (3.53), $\alpha \to \widehat\alpha$, $\alpha^* \to \widehat\alpha^\dagger$ (3.55) with $[\widehat\alpha(\vec p),\widehat\alpha^\dagger(\vec p')] = (2\pi)^3(2\omega_p)\delta(\vec p - \vec p')$ (3.27). $\widehat H = \int d^3p\,[\omega_p\widehat\alpha^\dagger\widehat\alpha + \frac12\omega_p\delta(\vec 0)]$ (3.56); normal ordering removes the vacuum energy (3.57), $E_{\rm vac} = \sum_{\vec p}\frac12\omega_p$ (3.58). "In the canonical quantization formalism the concept of particle appears as a result of the quantization of a classical field" (p.18). Since time is privileged, Lorentz invariance must be checked at the end (p.16).
- Renormalisation (sec. 8.1, p.80-83): $\mathcal O_{\rm physical} = \lim_{\Lambda\to\infty}[\mathcal O(\Lambda)_{\rm bare} + \Delta\mathcal O(\Lambda)_\hbar]$ (8.1). Vacuum polarisation gives $\Pi(q^2) \simeq \frac{e^2}{12\pi^2}\log(q^2/\Lambda^2)$ (8.10); the measurable charge at scale $\mu$ obeys $e(\mu)^2 = e(\Lambda)^2_{\rm bare}[1 + \frac{e(\Lambda)^2_{\rm bare}}{12\pi^2}\log(\mu^2/\Lambda^2)]$ (8.13) and, cutoff eliminated, $e(\mu)^2 = e(\mu_0)^2[1 + \frac{e(\mu_0)^2}{12\pi^2}\log(\mu^2/\mu_0^2)]$ (8.15, p.84). Renormalisability of (4.79) requires $V$ at most quartic (p.36).
- Anomalies (sec. 7, p.66-67): an anomaly is "a classical symmetry that cannot be realized in the quantum theory, no matter how smart we are in choosing the regularization procedure", not an artefact of a bad regulator. Axial current $J^\mu_A = \overline\psi\gamma^\mu\gamma_5\psi$ (7.5) acquires $\partial_\mu J^\mu_A \sim \hbar\,\vec E\cdot\vec B$ (7.6). Gauge anomalies cancel in the Standard Model between quarks and leptons (p.80, with the Witten SU(2) anomaly condition (7.60)); the gauge-anomaly coefficient is $\mathrm{Tr}[T^a_R\{T^b_R,T^c_R\}] = A(R)d^{abc}$ (A.62), vanishing for real or pseudoreal representations (A.63-A.64, p.110).

# Attributable statements

1. Relativistic invariance plus a single-particle interpretation leads to causality violation (probability leaks out of the light cone), which forces the introduction of quantum fields (sec. 2, p.10).
2. Canonical quantisation: Poisson brackets to commutators, $i\{\cdot,\cdot\} \to [\cdot,\cdot]$; the particle concept emerges from quantising the classical field (sec. 3.1, eqs 3.51, 3.55, p.17-18).
3. Noether's theorem: every continuous symmetry (one with $\delta_\epsilon\mathcal L = \partial_\mu K^\mu$ off-shell) yields a conserved current $J^\mu = \frac{\partial\mathcal L}{\partial(\partial_\mu\phi)}\delta_\epsilon\phi - K^\mu$ and a Lorentz-scalar charge that generates the symmetry after quantisation (sec. 6.1, eqs 6.3-6.8, p.57-58).
4. Promoting the global U(1) of the Dirac Lagrangian to a local symmetry "it suffices to replace the ordinary derivative by a covariant one", $D_\mu = \partial_\mu + ieA_\mu$ (sec. 4.3, eqs 4.68-4.69, p.34).
5. The non-abelian field strength (4.75) is not gauge invariant but transforms covariantly, $F_{\mu\nu} \to UF_{\mu\nu}U^{-1}$ (sec. 4.3, eq. 4.76, p.35).
6. Gauge symmetry "reflects a degree of redundancy in the description of physical states"; each physical state is an orbit under gauge transformations, which is why gauge fixing is necessary; in Maxwell theory the gauge transformations are generated by Gauss' law (sec. 4.4, eqs 4.94-4.96, p.38-39).
7. Goldstone's theorem: associated with each broken generator there is a massless mode; the mass matrix has as many zero modes as broken generators (sec. 6.2, eqs 6.38-6.39, p.62-63).
8. Distinct vacua are unreachable in the infinite-volume limit: $\langle 0_1|0_2\rangle = 0$ and no local observable connects them (sec. 6.2, p.63).
9. Higgs mechanism: "at no point we have given up gauge invariance. The symmetry is only hidden"; the photon mass is $m_\gamma^2 = 2e^2\mu^2$ and the Goldstone mode becomes the longitudinal polarisation (sec. 6.2, eq. 6.48, p.65).
10. Anomalies are not regulator artefacts: a classical symmetry that cannot survive quantisation however the theory is regularised; gauge anomalies cancel between quarks and leptons in the Standard Model (sec. 7, p.66-67, p.80).

# Caveats

- Coupling sign: the abelian section uses $D_\mu = \partial_\mu + ieA_\mu$ with $\psi \to e^{-ie\varepsilon}\psi$ (4.69, 4.71), the non-abelian section $D_\mu = \partial_\mu - igA^a_\mu T^a$ with $\Phi \to e^{i\chi^aT^a}\Phi$ (4.74, 4.78). Taking $G = $ U(1), $T = 1$, $g = e$ in (4.74) gives $A_\mu \to A_\mu + \frac1e\partial_\mu\chi$ with $\Phi \to e^{i\chi}\Phi$; matching to (4.71) requires $A_\mu^{\rm (4.78)} = -A_\mu^{\rm (4.69)}$. The tutorial should pick one sign and state the map.
- (6.44) in the Abelian Higgs model writes $\Phi \to e^{i\alpha(x)}\Phi$, $A_\mu \to A_\mu + \partial_\mu\alpha$ while using (4.69); with (4.69) the consistent pair is $\Phi \to e^{-ie\alpha}\Phi$, $A_\mu \to A_\mu + \partial_\mu\alpha$ (as in 4.71). The paper is sloppy about the factor $-e$ in the phase; the physics (6.46)-(6.48) is unaffected.
- (6.47) interaction coefficients: substituting (6.46) with $\vartheta = 0$ into (6.43), I get $e^2A_\mu A^\mu(\mu^2 + \sqrt2\mu\sigma + \frac12\sigma^2)$ and $\frac{\lambda}{4}(\sqrt2\mu\sigma + \frac12\sigma^2)^2 = \frac12\lambda\mu^2\sigma^2 + \frac{\lambda\mu}{2\sqrt2}\sigma^3 + \frac{\lambda}{16}\sigma^4$. The quadratic terms (hence $m_\gamma^2 = 2e^2\mu^2$ and $m_\sigma^2 = \lambda\mu^2$) agree with the paper; the printed cubic and quartic coefficients ($\lambda\mu\sigma^3$, $\frac{\lambda}{4}\sigma^4$, $e^2\mu A^2\sigma$, $e^2A^2\sigma^2$) do not match this expansion. Treat the interaction terms of (6.47) as unreliable; verify in the tutorial's own computation.
- Unitary gauge: the phase $\vartheta$ is removed by the gauge choice $\alpha = -\vartheta$; "by identifying the gauge parameter with the Goldstone excitation we have completely fixed the gauge and the Lagrangian (6.47) does not have any gauge symmetry left" (p.65). No $R_\xi$ or Lorentz-gauge treatment of the Higgs model is given; Lorentz gauge $\partial_\mu A^\mu = 0$ (4.63) is used only for the free photon, with the remark that it does not fix the gauge completely (p.33) and that negative-norm states must be shown to decouple (p.34).
- Potential form: SSB potentials are written as $\frac{\lambda}{4}(\vec\varphi^{\,2} - a^2)^2$ (6.40) and $\frac{\lambda}{4}(\overline\Phi\Phi - \mu^2)^2$ (6.43), with $\mu$ the VEV (not a mass parameter) and vacuum energy zero (footnote 14, p.61). Map to $-m^2|\phi|^2 + \lambda'|\phi|^4$: $\lambda' = \lambda/4$, $m^2 = \lambda\mu^2/2$, plus a constant.
- Missing items (do not cite this paper for them): Bianchi identity; infinitesimal non-abelian gauge transformation; $F_{\mu\nu}$ as $[D_\mu, D_\nu]$; non-abelian Higgs model, SU(2)xU(1) gauge-boson mass matrix, $W^\pm$/$Z^0$ masses (mentioned only in words, p.64, p.66); complex-scalar Noether current via (6.5) (only the Dirac case is worked out); $\mathrm{Tr}$-normalised Yang-Mills action (the paper sums over $a$ explicitly).
- Generator exponent sign differs between the main text ($U = \exp(i\alpha_aT^a)$, 6.18; $e^{i\chi^aT^a}$, 4.74) and the Appendix ($D(g) = \exp(-i\alpha_aT^a)$, A.46; $e^{-\frac{i}{2}\alpha_i\sigma^i}$, A.48). Structure constants are the same ($[T^a,T^b] = if^{abc}T^c$) in both.
- Minor typo: (3.46) prints $m^2$ for $m^2\phi^2$ in the Hamiltonian density.
- Page numbering: PDF page = printed page, so citations can use a single number.
