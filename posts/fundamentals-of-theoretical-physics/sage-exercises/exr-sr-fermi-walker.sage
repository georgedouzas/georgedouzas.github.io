# Exercise exr-sr-fermi-walker: Fermi-Walker transport of a spin vector over one revolution of a circular orbit with
# v = 0.8, expressed in the pure-boost frame; the angle is -2 pi (gamma - 1).
import numpy as np
eta = np.diag([1., -1, -1, -1])
def boost(v):
    v = np.array(v, float); b = np.linalg.norm(v); g = 1/np.sqrt(1 - b*b); B = np.eye(4)
    B[0, 0] = g; B[0, 1:] = g*v; B[1:, 0] = g*v
    if b > 0: B[1:, 1:] += (g - 1)*np.outer(v, v)/b**2
    return B
def angle_axis(R):
    R3 = R[1:, 1:]; ang = np.arccos(np.clip((np.trace(R3) - 1)/2, -1, 1))
    ax = np.array([R3[2, 1] - R3[1, 2], R3[0, 2] - R3[2, 0], R3[1, 0] - R3[0, 1]])
    return ang, ax/np.linalg.norm(ax) if np.linalg.norm(ax) > 1e-14 else ax

w0, r = 1.0, 0.8; v = w0*r; g = 1/np.sqrt(1 - v*v)
U = lambda t: g*np.array([1, -r*w0*np.sin(w0*t), r*w0*np.cos(w0*t), 0])
A = lambda t: g*g*np.array([0, -r*w0**2*np.cos(w0*t), -r*w0**2*np.sin(w0*t), 0])     # du/dtau
dot = lambda p, q: p @ eta @ q
S = np.array([0, 1.0, 0, 0]); T = 2*np.pi/w0; N = 100000; dt = T/N; t = 0.0
f = lambda t, S: -(dot(A(t), S))*U(t)/g          # dS/dt = (dS/dtau)/gamma
for _ in range(N):
    k1 = f(t, S); k2 = f(t + dt/2, S + dt/2*k1); k3 = f(t + dt/2, S + dt/2*k2); k4 = f(t + dt, S + dt*k3)
    S = S + dt/6*(k1 + 2*k2 + 2*k3 + k4); t += dt
Sr = np.linalg.inv(boost(U(T)[1:]/U(T)[0])) @ S
print("S.u = %.2e, S.S = %.10f" % (dot(S, U(T)), dot(S, S)))
print("spin in the boosted frame after one revolution:", np.round(Sr, 6))
ang = np.arctan2(Sr[2], Sr[1])
print("angle %.6f; -2 pi (gamma - 1) mod 2 pi = %.6f" % (ang, (-2*np.pi*(g - 1)) % (2*np.pi)))
