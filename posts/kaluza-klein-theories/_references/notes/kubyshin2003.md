# Identity

- arXiv: math-ph/0309059 (v1, 29 Sep 2003)
- Title: Geometrical formalism in gauge theories
- Author: Yuri A. Kubyshin (Dept. Matematica Aplicada IV, Univ. Politecnica de Cataluna, Barcelona; and Institute of Nuclear Physics, Moscow State University)
- Year: 2003
- Venue: footnote on p. 1: "Talk given at the CEDYA-03 Congress (Tarragona, September 15-19, 2003)". No journal stated.
- Length: 20 PDF pages; the PDF carries no printed page numbers, so all page references below are PDF page numbers as shown by the Read tool.

# Structure

1. Introduction — pp. 1-3 (unnumbered sections throughout; eqs. (1)-(9))
2. Geometrical formalism for description of gauge theories — pp. 3-5 (eqs. (10)-(12) plus several unnumbered displays)
3. Invariant connections and dimensional reduction — pp. 5-10 (eqs. (13)-(24))
4. Example of invariant connection — pp. 10-14 (eqs. (25)-(40)); SU(2) Yang-Mills on S^2 = SU(2)/U(1)
5. Classification of fibre bundles — pp. 14-18 (eqs. (41)-(52), Theorems 1 and 2)
6. Acknowledgements, References [1]-[38] — pp. 18-20

# Conventions

- Gauge group G: compact Lie group; Lie algebra denoted $\mathcal{G}$ (calligraphic). Adjoint actions (p. 1, eqs. (1), (2)): $h \to g h g^{-1}$ and $X \to ad(g)X \equiv g X g^{-1}$. Lower-case $ad(g)$ is the *group* adjoint action on the algebra. In the example section (p. 10) upper-case $Ad(X)Y \equiv [X,Y]$ is the algebra adjoint action. No structure constants and no index placement conventions appear anywhere in the paper.
- Generators: anti-Hermitian. Explicitly (p. 10) the generators of $K = SU(2)$ are $Q_j = \sigma_j/2i$. Gauge potentials $A_\mu$ are $\mathcal{G}$-valued (no coupling $e$ and no factor $i$ inside $A$); the coupling $e$ appears only as an overall $1/(8e^2)$ or $1/(4e^2)$ in the action. The covariant derivative is $D_\mu(A) = \partial_\mu + \rho'(A_\mu)$ (eq. (7), p. 2), with $\rho'$ the Lie-algebra representation induced by $\rho$.
- Gauge transformation of the potential (eq. (4), p. 2): $A'_\mu = ad(g(x)^{-1})A_\mu + g^{-1}\partial_\mu g$, i.e. $A' = g^{-1} A g + g^{-1} dg$. Note the paper uses $ad(g^{-1})$ rather than $ad(g)$; this is the "pull-back by a change of section $s \to \Psi_g s$" convention, consistent with $w_0 = \varphi^{-1} d\varphi$.
- Field strength (eq. (9), p. 2): $F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu - [A_\mu, A_\nu]$ — note the MINUS sign on the commutator in components. In form language the paper writes (p. 9, below eq. (24)) $F = dA + [A, A]$ and (p. 12, before eq. (36)) $F = dA + \tfrac12 [A, A]$, and for the connection curvature (p. 9) $\hat\Omega = d\hat w + \tfrac12[\hat w, \hat w]$ and (p. 4) $\Omega = Dw$. With $A = A_\mu dx^\mu$ and $F = \tfrac12 F_{\mu\nu} dx^\mu\wedge dx^\nu$ (eq. (11), p. 3), $\tfrac12[A,A] = A\wedge A = \tfrac12 [A_\mu, A_\nu] dx^\mu \wedge dx^\nu$, so $F = dA + \tfrac12[A,A]$ gives $F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu + [A_\mu, A_\nu]$, which has the OPPOSITE sign on the commutator relative to eq. (9). See Caveats: the paper is internally inconsistent in the sign/factor of the commutator term. The explicit S^2 computation (eq. (36)) is consistent with $F = dA + A\wedge A = dA + \tfrac12[A,A]$.
- Yang-Mills action (eq. (8), p. 2): $S_{YM}[A] = \int_M \sqrt{|\det\gamma_{\mu\nu}|}\, d^Dx\, \frac{1}{8e^2}\langle F_{\mu\nu}F^{\mu\nu}\rangle$, where $\langle\cdot,\cdot\rangle$ is the Killing form in $\mathcal{G}$; in form language (eq. (10), p. 3) $S_{YM}[A] = \int_M \frac{1}{4e^2}\langle F\wedge *F\rangle$ with $*$ the Hodge star of the metric $\gamma$ on $M$. The relative factor 2 between (8) and (10) follows from $F = \tfrac12 F_{\mu\nu}dx^\mu\wedge dx^\nu$. Signature/sign of the Killing form is not specified.
- Principal bundle: $P = P(M, G)$, free right action $\Psi_g$ of $G$ on $P$, projection $\pi$; local trivialization $\chi: \pi^{-1}(U)\to U\times G$, $\chi(p) = (\pi(p), \varphi(p))$ with $\varphi(\Psi_g p) = \varphi(p) g$ (p. 4, unnumbered). The connection is a $\mathcal{G}$-valued 1-form $w$ on $P$ (the paper writes $w$, not $\omega$); the vertical isomorphism $\sigma: \mathcal{G} \to$ vertical vector fields is induced by $\Psi_g$.
- Coset space conventions: $M = M_4 \times K/H$, $K$ and $H$ compact; $K$ acts on $K/H$ by $y = [k_1] \mapsto ky = [k k_1]$ (left cosets, left action); reductive decomposition $\mathcal{K} = \mathcal{H}\oplus\mathcal{M}$ with $ad(H)\mathcal{M}\subset\mathcal{M}$ (eq. (18), p. 7). $\theta$ is the canonical left-invariant Maurer-Cartan form on $K$ ($\bar\theta = k^{-1}dk$ in the example, eq. (28), p. 11).

# Equations the tutorial uses

## (a) Principal bundle, local sections, transition functions (pp. 3-5)

- Definition (p. 3-4, no eq. number): $P = P(M,G)$ with base $M$ (space-time) and structure group $G$ (gauge group); free action $\Psi_g: P\to P$, projection $\pi: P\to M$. Local triviality: every $x\in M$ has an open $U\ni x$ with $\pi^{-1}(U)\cong U\times G$ via $\chi(p) = (\pi(p), \varphi(p))$, $\varphi: \pi^{-1}(U)\to G$, $\varphi(\Psi_g p) = \varphi(p)g$.
- Local section (p. 4): $s: U \to \pi^{-1}(U)\subset P$; "Choosing a family of sections for an open covering of $M$ corresponds to choosing a gauge condition."
- Canonical section attached to a trivialization (p. 4): for fixed $a\in G$, $s_a(x) = \chi^{-1}(x, a)$.
- Change of section (p. 4): if $t(x) = \Psi_{g(x)} s(x)$ then
  $$A^t_\mu(x) = ad(g(x)^{-1})A^s_\mu(x) + g^{-1}(x)\partial_\mu g(x) \quad\text{(unnumbered, p. 4)}.$$
- Transition functions in the usual sense are NOT defined in this paper; the gluing is handled by the change-of-section formula above and (in the example) by explicit gauge transformations $V_1, V_2$ (p. 13). Global statement (p. 5): "Non-existence of a global cross section of a non-trivial fibre bundle $P(M,G)$ means the impossibility to introduce the gauge fixing globally... Contrary to this, the connection 1-form $w$ exists globally."

## (b) Connection 1-form, local gauge potential, gauge transformation (pp. 2-4)

- Gauge 1-form: $A = A_\mu dx^\mu$ per chart (p. 3). Gauge transformation in form language (p. 3, unnumbered):
  $$A \to A' = ad(g^{-1})A + g^{-1}dg .$$
- Component form, eq. (4), p. 2: $A_\mu(x)\to A'_\mu(x) = ad(g(x)^{-1})A_\mu(x) + g^{-1}(x)\partial_\mu g(x)$; matter: eq. (5) $\psi' = \rho(g(x))\psi$.
- Pull-back relation (p. 4): $A^{(s)} = s^* w$ and $F^{(s)} = s^*\Omega$, $\Omega = Dw$ the curvature 2-form of $w$.
- Local expression of the connection form, eq. (12), p. 4:
  $$w = w_0 + \sigma(A^{s_a}_\mu(x))\,\bar{d}x^\mu ,$$
  where $\bar{d}x^\mu$ is the 1-form on $\pi^{-1}(U)$ dual to the lifted coordinate vector fields $\bar\xi_\mu$ (tangent to $\chi^{-1}(U,a)$), $\bar{d}x^\mu(\bar\xi_\nu) = \delta^\mu_\nu$, and $w_0 = \varphi^*\theta$ with $\theta$ the canonical left-invariant form on $G$; for matrix groups (p. 4, unnumbered)
  $$w_0 = \varphi(p)^{-1} d\varphi(p).$$
- Horizontal lift (p. 4, unnumbered): $\xi^*_\mu = \bar\xi_\mu - \sigma(A^{s_a}_\mu(x))$.

## (c) Curvature, Bianchi identity, field-strength components

- Components, eq. (9), p. 2: $F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu - [A_\mu, A_\nu]$.
- 2-form, eq. (11), p. 3: $F = \tfrac12 F_{\mu\nu}dx^\mu\wedge dx^\nu$.
- Curvature of the connection (p. 9, text below eq. (23)): $\hat\Omega = d\hat w + \tfrac12[\hat w,\hat w]$; and $F = dA + [A,A]$ (p. 9, text below eq. (24)); and $F = dA + \tfrac12[A,A]$ (p. 12, text above eq. (36)).
- Bianchi identity: NOT stated anywhere in the paper.
- Covariant derivative of the scalar (p. 9, unnumbered, "cf. (7)"): $(D\phi)_x(\bar\theta_\mathcal{M}) = d\phi_x(\bar\theta_\mathcal{M}) + [A_x, \phi_x(\bar\theta_\mathcal{M})]$.

## (d) Associated bundles and matter fields

- Associated bundles are NOT introduced. Matter fields appear only in the "physical" formalism: $\psi(x)\to\rho(g)\psi(x)$ (eq. (3), p. 2), local version eq. (5), total Lagrangian eq. (6), p. 2:
  $$\mathcal{L}(\psi,\partial_\mu\psi, A) = \mathcal{L}_m(\psi, D_\mu(A)\psi) + \mathcal{L}_{YM}(A), \qquad D_\mu(A) = \partial_\mu + \rho'(A_\mu) \ (7).$$

## (e) Symmetric gauge fields and dimensional reduction (pp. 5-10)

- Setting (p. 5): $M = M_4\times K/H$, bundle $\hat P = P(M, G)$. $K$ acts on $M$ by $O_k(x,y) = (x, ky)$, trivially on $M_4$. Assumed lift to $\hat P$ as a left bundle automorphism $L_k$ with
  $$L_{k_1k_2} = L_{k_1}\circ L_{k_2} \ (13), \qquad L_k\circ\Psi_g = \Psi_g\circ L_k \ (14), \qquad \pi\circ L_k = O_k\circ\pi \ (\text{p. 6, unnumbered}).$$
  Footnote 1 (p. 5): the existence of the lift is usually taken as the starting assumption; the lifting problem is not addressed.
- Definition of $K$-invariant connection, eq. (15), p. 6: $L_k^*\hat w = \hat w$. $K$-invariant metric, eq. (16): $O_k^*\hat\gamma = \hat\gamma$; then the gauge theory is called $K$-invariant.
- Symmetric gauge field, eq. (17), p. 6: for a local section $s: U\to\hat P$ and $\hat A = s^*\hat w$,
  $$O^*_{k^{-1}}\hat A = ad(\rho_k)^{-1}\hat A + (\rho_k)^{-1}d\rho_k ,$$
  where $\rho_k: U\to G$ is defined by $L_k s(\hat x) = \Psi_{\rho_k(O_k\hat x)} s(O_k\hat x)$ (p. 6, unnumbered). "Gauge fields satisfying relation (17) are called symmetric gauge fields" (first introduced in [37] Witten 1977 and [29] Schwarz / Romanov-Schwarz-Tyupkin 1977). The action of $K$ on a symmetric gauge field is equivalent to a gauge transformation; hence $\mathcal{L}_{YM}$ on such fields is independent of $y\in K/H$ and is effectively a Lagrangian on $M_4$ — the basis of CSDR (p. 6).
- Reductive decomposition, eq. (18), p. 7: $\mathcal{K} = \mathcal{H}\oplus\mathcal{M}$, $ad(H)\mathcal{M}\subset\mathcal{M}$, which implies $[\mathcal{H},\mathcal{M}]\subset\mathcal{M}$.
- Isotropy homomorphism, eq. (19), p. 7: on $\tilde P = \pi^{-1}(\tilde M)$, $\tilde M = M_4\times\{o\}$, $L_h\tilde p = \Psi_{\tau_{\tilde p}(h)}\tilde p$ defines $\tau_{\tilde p}: H\to G$, a homomorphism.
- $H$-invariant restricted form, eq. (20), p. 7: $\tilde w = i_e^*\hat w$ satisfies $L_h^*\tilde w = \tilde w$.
- Equivariant map, p. 7: $\alpha_{\tilde p} \equiv i_{\tilde p}^*\hat w = \Lambda_{\tilde p}(\theta)$ with $i_{\tilde p}(k) = L_k\tilde p$, $\Lambda_{\tilde p}: \mathcal{K}\to\mathcal{G}$ linear, and (eq. (21), p. 7)
  $$\Lambda_{\tilde p}\circ ad(h) = ad(\tau_{\tilde p})\circ\Lambda_{\tilde p}.$$
  Restrictions: $\Lambda_{\tilde p}|_\mathcal{H} = \tau'_{\tilde p}: \mathcal{H}\to\mathcal{G}$ (the differential of $\tau$), and $\Lambda_{\tilde p}|_\mathcal{M}\equiv\phi_{\tilde p}$ with the constraint, eq. (22), p. 7:
  $$\phi_{\tilde p}\circ ad(h) = ad(\tau_{\tilde p})\circ\phi_{\tilde p}.$$
- Reduction of the structure group (p. 8): $\tau_{\tilde p'}(h) = g^{-1}\tau_{\tilde p}(h)g$; fix $\tilde p_0$, $\tau\equiv\tau_{\tilde p_0}$; $P = \{p\in\tilde P: \tau_p = \tau\}$ is a principal bundle $P(M_4, C)$ with $C = C_G(\tau(H)) \equiv \{c\in G \mid c\tau(h)c^{-1} = \tau(h)\ \forall h\in H\}$, the centralizer; $w = \tilde w|_P$ is a $\mathcal{C}$-valued connection on $P$.
- Main classification result (Wang's theorem, p. 8, stated in words): "Any $K$-invariant connection form $\hat w$ on $\hat P(M_4\times K/H, G)$ is in one-to-one correspondence with the pair $(w, \phi_p)$ in the reduced bundle $P(M, C)$, where $w$ is an arbitrary, i.e. free of additional constraints, connection 1-form in $P$ and $\phi_p$ is a linear equivariant mapping $\phi_p: \mathcal{M}\to\mathcal{G}$ satisfying property (22)."
- Local form of the symmetric gauge field (p. 8): section $s(x,y) = L_{s_1(y)}s_2(x)$ with $s_1: U_{K/H}\to K$, $s_2: U_{M_4}\to P$; then $\hat A(x,y) \equiv (s^*\hat w)_{(x,y)} = (s_2^*w)_x + \Lambda_{s_2(x)}((s_1^*\theta)_y)$, i.e. eq. (23), p. 8:
  $$\hat A(x,y) = A(x) + \tau'(\bar\theta_\mathcal{H}) + \phi_x(\bar\theta_\mathcal{M}),$$
  with $A = s_2^*w$, $\bar\theta = s_1^*\theta$, $\phi_x = \phi_{s_2(x)}$, and $\bar\theta_{\mathcal{H}}, \bar\theta_{\mathcal{M}}$ the components w.r.t. (18). "Formula (23) describes a most general $K$-symmetric gauge field on $M$. This is basically Wang's result" (p. 9).
- Reduced field strength, eq. (24), p. 9:
  $$\hat F = s^*\hat\Omega = F + (D\phi)(\bar\theta_\mathcal{M}) + \tfrac12[\phi(\bar\theta_\mathcal{M}), \phi(\bar\theta_\mathcal{M})] - \tfrac12\phi\big([\bar\theta_\mathcal{M},\bar\theta_\mathcal{M}]_\mathcal{M}\big) - \tfrac12\tau'\big([\bar\theta_\mathcal{M},\bar\theta_\mathcal{M}]_\mathcal{H}\big),$$
  with $F = dA + [A,A]$ (as printed) on $M_4$, $(D\phi)_x(\bar\theta_\mathcal{M}) = d\phi_x(\bar\theta_\mathcal{M}) + [A_x, \phi_x(\bar\theta_\mathcal{M})]$, and $[\cdot,\cdot]_\mathcal{M}, [\cdot,\cdot]_\mathcal{H}$ the projections of the $\mathcal{K}$-bracket.
- Interpretation (p. 9): the reduced theory on $M_4$ has gauge group $C\subset G$, a gauge field $A$, and scalar fields generated by $\phi_x$; solving constraint (22) is "the second, algebraic part of the CSDR": decompose $\mathcal{M}$ under $H$ and $\mathcal{G}$ under $\tau(H)$ into irreducibles and use Schur's lemma to build the most general intertwiner. For symmetric $K/H$ the scalar potential is of Higgs type [23], [25]. Also noted: invariant connections can equivalently be defined via Lie derivatives along Killing vectors [20].

## (f) Explicit example: SU(2) Yang-Mills on $S^2 = SU(2)/U(1)$ (pp. 10-14; from Ref. [1])

- Charts eqs. (25), (26): $U_1 = \{0\le\vartheta<\pi\}$, $U_2 = \{0<\vartheta\le\pi\}$, $0\le\varphi<2\pi$.
- Generators $Q_j = \sigma_j/2i$; $\mathcal{H} = \mathrm{span}(Q_3)$, $\mathcal{M} = \mathrm{span}(Q_1, Q_2)$. Complexification: $e_{\pm\alpha} = \sigma_\pm = \tfrac12(\sigma_1\pm i\sigma_2)$, $h_\alpha = \sigma_3$, $[h_\alpha, e_{\pm\alpha}] = \pm2e_{\pm\alpha}$; decomposition eq. (27): $\underline{2}\to(2)+(-2)$.
- Local coset representatives (p. 10): $k^{(1)}(\vartheta,\varphi) = e^{-i\varphi\frac{\sigma_3}{2}}e^{i\vartheta\frac{\sigma_2}{2}}e^{i\varphi\frac{\sigma_3}{2}}$, $k^{(2)}(\vartheta,\varphi) = e^{i\varphi\frac{\sigma_3}{2}}e^{i(\vartheta-\pi)\frac{\sigma_2}{2}}e^{-i\varphi\frac{\sigma_3}{2}}$, viewed as local sections of $K = P(K/H, H)$.
- Maurer-Cartan components, eqs. (28)-(29), p. 11: $\bar\theta^{(i)} = (k^{(i)})^{-1}dk^{(i)} = \bar\theta^{(i)}_\mathcal{H} + \bar\theta^{(i)}_\mathcal{M}$,
  $$\bar\theta^{(1)}_\mathcal{H} = i\tfrac{\sigma_3}{2}(1-\cos\vartheta)d\varphi,\quad \bar\theta^{(1)}_\mathcal{M} = i\tfrac{\sigma_1}{2}(-\sin\varphi\, d\vartheta - \sin\vartheta\cos\varphi\, d\varphi) + i\tfrac{\sigma_2}{2}(\cos\varphi\, d\vartheta - \sin\vartheta\sin\varphi\, d\varphi),$$
  $$\bar\theta^{(2)}_\mathcal{H} = -i\tfrac{\sigma_3}{2}(1+\cos\vartheta)d\varphi,\quad \bar\theta^{(2)}_\mathcal{M} = i\tfrac{\sigma_1}{2}(\sin\varphi\, d\vartheta - \sin\vartheta\cos\varphi\, d\varphi) + i\tfrac{\sigma_2}{2}(\cos\varphi\, d\vartheta + \sin\vartheta\sin\varphi\, d\varphi).$$
- Homomorphisms $\tau_n: U(1)\to SU(2)$, $\tau(e^{i\frac{\sigma_3}{2}\alpha_3}) = e^{i\kappa\frac{\sigma_3}{2}\alpha_3}$, consistent iff $\kappa = n\in\mathbb{Z}$; eq. (30): $\tau'_n(h_\alpha) = nH_\alpha$; eq. (31): $\underline{3}\to(0)+(2n)+(-2n)$.
- $n\ne\pm1$ (and $n = 0$): $\phi = 0$ and eq. (32): $A^{(1)}_n = \tfrac{i}{2}n\sigma_3(1-\cos\vartheta)d\varphi$, $A^{(2)}_n = -\tfrac{i}{2}n\sigma_3(1+\cos\vartheta)d\varphi$ (Dirac-type monopoles with number $\kappa = n$).
- $n = 1$: eq. (33) $\phi(e_\alpha) = f_1E_\alpha$, $\phi(e_{-\alpha}) = f_2E_{-\alpha}$, reality condition $f_1 = f_2^*$ [24]; one complex parameter $f$. Potentials eqs. (34), (35), p. 12:
  $$A^{(1)}_1 = i\begin{pmatrix}(1-\cos\vartheta)d\varphi & fe^{-i\varphi}(-id\vartheta - \sin\vartheta d\varphi)\\ f^*e^{i\varphi}(id\vartheta - \sin\vartheta d\varphi) & -(1-\cos\vartheta)d\varphi\end{pmatrix},\quad
  A^{(2)}_1 = i\begin{pmatrix}-(1+\cos\vartheta)d\varphi & fe^{i\varphi}(-id\vartheta - \sin\vartheta d\varphi)\\ f^*e^{-i\varphi}(id\vartheta - \sin\vartheta d\varphi) & (1+\cos\vartheta)d\varphi\end{pmatrix}.$$
- Curvature eq. (36): $F = -i\sigma_3(|f|^2-1)\sin\vartheta\, d\vartheta\wedge d\varphi$ (globally defined). Action eq. (37): $\mathcal{S}_{YM}(f) = \frac{\pi}{2e^2}\frac{1}{R^2}(|f|^2-1)^2$; extrema at $f = 0$ (maximum, $\kappa = 1$ monopole) and $|f| = 1$ (minima, pure gauge, $\kappa = 0$). "Due to the $K$-invariance any extrema of the action found within the subspace of invariant connections is also an extremum in the space of all connections [24]" (p. 12).
- Gauge-equivalence of the patches (p. 13): $V_1 = i\begin{pmatrix}\cos\frac\vartheta2 & e^{-i\varphi}\sin\frac\vartheta2\\ e^{i\varphi}\sin\frac\vartheta2 & -\cos\frac\vartheta2\end{pmatrix}$, $V_2 = i\begin{pmatrix}e^{i\varphi}\cos\frac\vartheta2 & \sin\frac\vartheta2\\ \sin\frac\vartheta2 & -e^{-i\varphi}\cos\frac\vartheta2\end{pmatrix}$, $A^{(i)\prime}_1 = V_i^{-1}A^{(i)}_1V_i - V_i^{-1}dV_i$, giving the single global form eqs. (38)-(39) and, at $f = 0$, eq. (40) (the standard $\kappa = 1$ SU(2) monopole). Classes of gauge-inequivalent invariant connections are labelled by $|f|$ (p. 14).

## (g) Kaluza-Klein picture as a bundle

- Not formulated as a bundle over $M_4$ with fibre $K/H$ and KK gauge fields from the metric. The KK hypothesis is stated in words only (p. 5): extra dimensions compact of size $R<10^{-17}$ cm; the geometric object is the principal bundle $\hat P = P(M_4\times K/H, G)$ for the gauge field, and $K = P(K/H, H)$ for the coset space itself (p. 10). Reference [23] (Kubyshin-Mourao-Volobujev) is cited for Einstein-Yang-Mills reduction and spontaneous compactification.
- Topology-related result used in the example: eq. (43) $\mathcal{B}_G(S^n)\cong\pi_{n-1}(G)$, so $\mathcal{B}_{SU(2)}(S^2) = 0$: all SU(2) monopoles live on the trivial bundle $S^2\times SU(2)$ (pp. 12, 15). Eq. (45): $\mathcal{B}_{U(1)}(M)\cong H^2(M;\mathbb{Z})$; Theorem 1 (eq. (46), p. 16) and eqs. (47)-(49) for 2-dimensional $M$.

# Attributable statements

1. "Choosing a family of sections for an open covering of $M$ corresponds to choosing a gauge condition." (Sec. Geometrical formalism, p. 4)
2. "Non-existence of a global cross section of a non-trivial fibre bundle $P(M,G)$ means the impossibility to introduce the gauge fixing globally... Contrary to this, the connection 1-form $w$ exists globally. It contains all the information about the gauge field configuration." (p. 5)
3. The lift $L_k$ of the $K$-action on $M$ to a bundle automorphism is assumed; "in many mathematical studies and physical applications the existence of a group bundle automorphisms $L_k$ with properties (13), (14) is taken as the initial assumption, so that the problem of the lift of $O_k$ is not addressed." (footnote 1, p. 5)
4. Symmetric gauge fields, defined by eq. (17) (invariance of $\hat A$ under $K$ up to a gauge transformation with compensating $\rho_k$), were first introduced by Witten [37] and Schwarz / Romanov-Schwarz-Tyupkin [29], who also classified static spherically-symmetric fields in 3 dimensions. (p. 6)
5. Because the $K$-action on symmetric gauge fields is a gauge transformation and $\mathcal{L}_{YM}$ is gauge invariant, the Lagrangian evaluated on symmetric fields is independent of the point of $K/H$ and is an effective Lagrangian on $M_4$ — "the basis of the method of dimensional reduction... called the coset space dimensional reduction (CSDR)." (p. 6)
6. Wang's theorem as stated: a $K$-invariant connection on $\hat P(M_4\times K/H, G)$ is in one-to-one correspondence with a pair $(w, \phi)$: an unconstrained connection on the reduced bundle $P(M_4, C)$, $C = C_G(\tau(H))$, and a linear $H$-equivariant map $\phi: \mathcal{M}\to\mathcal{G}$ (eq. (22)). (p. 8; Wang [35], Kobayashi-Nomizu [21]; physics: Forgacs-Manton [13], Harnad-Shnider-Vinet [18]; reviews [24], [20])
7. The reduced theory on $M_4$ has gauge group $C\subset G$, gauge field $A$ and scalars from $\phi_x$; solving the constraint (22) is the "second, algebraic part of the CSDR", done via Schur's lemma after decomposing $\mathcal{M}$ under $H$ and $\mathcal{G}$ under $\tau(H)$. (p. 9)
8. For symmetric homogeneous spaces $K/H$ the scalar potential of the reduced theory is always of Higgs type and leads to spontaneous symmetry breaking [23], [25]. (p. 9)
9. Any extremum of the action within the subspace of invariant connections is an extremum in the space of all connections (principle of symmetric criticality, attributed to [24]). (p. 12)
10. All SU(2) monopoles on $S^2$ are reproduced by invariant connections labelled by $\tau_n$, live on the trivial bundle (since $\pi_1(SU(2)) = 0$), and the $n = \pm1$ sector contains a 1-parameter family interpolating between the $\kappa = \pm1$ and $\kappa = 0$ monopoles. (pp. 13-15)

# Caveats

- Commutator sign/factor inconsistency: eq. (9) has $F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu - [A_\mu, A_\nu]$ (minus sign), while the form-language curvature is written as $\hat\Omega = d\hat w + \tfrac12[\hat w,\hat w]$ (p. 9), $F = dA + [A,A]$ (p. 9, without the 1/2 — almost certainly a typo for $\tfrac12[A,A]$), and $F = dA + \tfrac12[A,A]$ (p. 12). With $A = A_\mu dx^\mu$ and (11), $dA + \tfrac12[A,A]$ has $+[A_\mu, A_\nu]$. The eq. (9) minus sign would be consistent with the gauge law (4) only if one took $A_\mu$ to be minus the connection pulled back, which the paper does not do. Checked symbolically (sympy) on the $S^2$ example: building $A^{(1)}_1$ from eqs. (23), (28), (30), (33) (which gives the matrix of eq. (34) with overall prefactor $i/2$, not $i$) and computing $F_{\vartheta\varphi} = \partial_\vartheta A_\varphi - \partial_\varphi A_\vartheta \pm [A_\vartheta, A_\varphi]$: the PLUS sign gives the diagonal result $-\tfrac{i}{2}\sigma_3(|f|^2-1)\sin\vartheta$ (eq. (36) up to a factor 2), while the MINUS sign of eq. (9) leaves non-vanishing off-diagonal terms $\propto f(1-\cos\vartheta)e^{\mp i\varphi}$ and is therefore wrong. The tutorial should use $F = dA + A\wedge A$, $F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu + [A_\mu, A_\nu]$ with anti-Hermitian $A$, and not cite eq. (9) verbatim.
- Factor-2 misprints in the example: eq. (34)/(35) as printed have overall prefactor $i$, but at $f=0$ they should reduce to eq. (32) at $n=1$, which has $\tfrac{i}{2}$; eqs. (23)+(28) also give $\tfrac{i}{2}$. With the printed prefactor $i$ neither commutator sign yields a curvature proportional to $\sigma_3$. With the corrected prefactor $\tfrac{i}{2}$ the curvature is $-\tfrac{i}{2}\sigma_3(|f|^2-1)\sin\vartheta\,d\vartheta\wedge d\varphi$, half of eq. (36). Eq. (37) inherits whatever normalization of the Killing form is used, so its overall coefficient should not be quoted without recomputation. The qualitative conclusions (extrema at $f=0$ and $|f|=1$, globally defined $F$) are unaffected.
- Gauge-law convention: the paper's transformation $A' = g^{-1}Ag + g^{-1}dg$ (eq. (4)) is the "passive" one arising from changing the section by $s\to\Psi_g s$; many texts write $A' = gAg^{-1} + g\,dg^{-1}$ or $-dg\,g^{-1}$. In the example the patch transformation is written $A' = V^{-1}AV - V^{-1}dV$ (p. 13), i.e. with the opposite sign on the inhomogeneous term relative to eq. (4). Checked: $V_1^2 = -1$ (so $V_1^{-1} = -V_1$ and $-V_1^{-1}dV_1 = V_1 dV_1$), but $V_2^2 \ne -1$ (both are unitary), so this is not simply a rewriting of eq. (4); the tutorial should recompute the patching rather than quote.
- No Bianchi identity, no transition functions $g_{ij}$, no associated bundle, no horizontal/vertical splitting as a definition of connection (only via the explicit formula (12)), no structure constants. The paper relies on Kobayashi-Nomizu [21] and Isham [19] for the general definitions.
- Reductive decomposition: eq. (18) lists $ad(H)\mathcal{M}\subset\mathcal{M}$ as a property of a reductive homogeneous space and asserts it holds whenever $K$, $H$ are compact (true: use an $ad(K)$-invariant inner product).
- The map $\Lambda_{\tilde p}$ depends on the point $\tilde p$; the paper suppresses this for $\tau$ after fixing $\tilde p_0$ and reducing to $P(M_4, C)$, but keeps $\phi_{\tilde p}$ (becoming $\phi_x$) point-dependent — this is what makes $\phi$ a scalar field on $M_4$.
- Eq. (23) is called "basically Wang's result" but is really the local (section-dependent) form; the sections $s_1, s_2$ must satisfy "certain natural conditions" (p. 8) that are not spelled out.
- Eq. (24): $F$ in the text is written $dA + [A,A]$ — see first caveat. The term $-\tfrac12\tau'([\bar\theta_\mathcal{M},\bar\theta_\mathcal{M}]_\mathcal{H})$ uses $\tau'$ on $\mathcal{H}$; this and the $\phi([\cdot,\cdot]_\mathcal{M})$ term vanish for symmetric spaces only in the $\mathcal{M}$-projection ($[\mathcal{M},\mathcal{M}]\subset\mathcal{H}$), not the $\mathcal{H}$-projection.
- $\pi_1(SO(n)) = Z_2$ is listed for "$n=3$ and $n\ge5$" (p. 17); $SO(4)$ also has $\pi_1 = Z_2$ but is omitted, presumably because it is not simple.
- The PDF has no printed page numbers; all page references here are PDF page indices.
- The first-page sentence "the gauge theories iimensional base spaces... problem of classificatimportant" is garbled in the source PDF; the intended meaning is that invariant connections are used to describe gauge theories on multidimensional base spaces and that bundle classification matters for the functional-integral formulation.
