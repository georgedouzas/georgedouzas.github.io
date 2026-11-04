# Exercise exr-sr-opening-angle (Boudreau-Swanson, Exercise 14.6): isotropic two-body decays boosted to the laboratory;
# invariant mass check and mean opening angle of B0 -> J/psi K*0.
import numpy as np
rng = np.random.default_rng(int(2))
M, m1, m2 = 5280.0, 3097.0, 892.0
eta = np.diag([1., -1, -1, -1])
def boost(v):
    b = np.linalg.norm(v); g = 1/np.sqrt(1 - b*b); B = np.eye(4)
    B[0, 0] = g; B[0, 1:] = g*v; B[1:, 0] = g*v
    if b > 0: B[1:, 1:] += (g - 1)*np.outer(v, v)/b**2
    return B
def decay(P):
    Mp = np.sqrt(P @ eta @ P); E1 = (Mp**2 + m1**2 - m2**2)/(2*Mp); q = np.sqrt(E1**2 - m1**2)
    c = rng.uniform(-1, 1); ph = rng.uniform(0, 2*np.pi); n = np.array([np.sqrt(1 - c*c)*np.cos(ph), np.sqrt(1 - c*c)*np.sin(ph), c])
    p1 = np.concatenate([[E1], q*n]); p2 = np.concatenate([[Mp - E1], -q*n])
    B = boost(P[1:]/P[0]); return B @ p1, B @ p2
for ratio in [1.5, 5.0]:
    E = ratio*M; P = np.array([E, 0, 0, np.sqrt(E**2 - M**2)])
    angs, dev = [], 0
    for _ in range(4000):
        a, b = decay(P); dev = max(dev, abs(np.sqrt((a + b) @ eta @ (a + b)) - M))
        angs.append(np.arccos(a[1:] @ b[1:]/np.linalg.norm(a[1:])/np.linalg.norm(b[1:])))
    print("E/M = %.1f: mean opening angle %.3f rad, 2/gamma = %.3f, max |m(p1+p2) - M| = %.1e MeV" % (ratio, np.mean(angs), 2/ratio, dev))
