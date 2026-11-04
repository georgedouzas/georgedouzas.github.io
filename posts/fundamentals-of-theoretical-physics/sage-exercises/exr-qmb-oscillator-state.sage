# Exercise exr-qmb-oscillator-state (Cini et al., Sec. 3.6, Problem 2): the oscillator state
# psi(0) = (sqrt(2)/3)|0> + i (sqrt(6)/3)|1> - (1/3)|2> in units hbar = m = omega = 1; norm, energy probabilities,
# mean energy, and the time dependence of <x> and <x^2> in the truncated number basis (exact here).
N = 6
a = matrix(SR, N, N, lambda i, j: sqrt(j) if i == j - 1 else 0)   # annihilation operator a|n> = sqrt(n)|n-1>
X = (a + a.transpose()) / sqrt(2)
t = var('t'); assume(t, 'real')
c0 = vector(SR, [sqrt(2) / 3, I * sqrt(6) / 3, -1 / 3] + [0] * (N - 3))
print("norm squared:", sum(abs(z)^2 for z in c0).simplify_full())
print("probabilities of E = 1/2, 3/2, 5/2:", [ (abs(c0[n])^2).simplify_full() for n in range(3)])
print("mean energy:", sum(abs(c0[n])^2 * (n + 1 / 2) for n in range(3)).simplify_full())
ct = vector(SR, [c0[n] * exp(-I * (n + 1 / 2) * t) for n in range(N)])
conj = vector(SR, [z.conjugate() for z in ct])
xm = (conj * X * ct).simplify_full()
x2 = (conj * X * X * ct).simplify_full()
print("<x>(t) =", xm.rectform().simplify_full().simplify_trig())
print("<x^2>(t) =", x2.rectform().simplify_full().simplify_trig().expand())
