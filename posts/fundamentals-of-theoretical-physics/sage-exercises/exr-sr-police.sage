# Exercise exr-sr-police (Blennow-Ohlsson, Problem 1.47): a pursuit with constant proper acceleration a of a ship moving
# at speed v; meeting time in three frames and the relative speed at the end (a = 1).
for v in [0.3, 0.6, 0.9]:
    g = 1/sqrt(1 - v^2); T = 2*v*g^2
    resid = v*T - (sqrt(1 + T^2) - 1)
    V = T/sqrt(1 + T^2)
    print("v = %.1f: meeting at t = %.4f (residual %.1e), crew %.4f, officer %.4f, officer's speed %.4f = 2v/(1+v^2) = %.4f, relative speed %.4f"
          % (v, T, resid, T/g, arcsinh(T), V, 2*v/(1 + v^2), (V - v)/(1 - V*v)))
