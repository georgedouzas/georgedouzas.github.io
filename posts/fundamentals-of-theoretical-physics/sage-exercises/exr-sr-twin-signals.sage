# Exercise exr-sr-twin-signals: birthday signals exchanged by the twins of the twin paradox (v = 4/5, turn after 3 years
# of the traveller's time); light signals followed in the Earth frame (years, light-years).
v = 4/5; g = 1/sqrt(1 - v^2); Tturn = 3*g; X = v*Tturn
def traveller(t):            # position of the traveller at Earth time t
    return v*t if t <= Tturn else X - v*(t - Tturn)
# Earth signals sent at t = 1..10; received by the traveller when t - tsend = traveller(t)
recv = []
for k in range(1, 11):
    t_out = k/(1 - v)                     # catch the outgoing traveller: t - k = v t
    t = t_out if t_out <= Tturn else (k + X + v*Tturn)/(1 + v)   # else meet on the way back: t - k = X - v (t - Tturn)
    tau = t/g
    recv.append(tau)
print("Earth's signals received at traveller's ages (years after departure):", [round(r, 3) for r in recv])
# traveller's signals at her proper times 1..6
earth = []
for k in range(1, 7):
    t = k*g; x = traveller(t)
    earth.append(t + x)
print("traveller's signals received on Earth at:", [round(e, 3) for e in earth])
print("received before reunion (t = 10):", sum(1 for e in earth if e <= 10 + 1e-12), "and", sum(1 for r in recv if r <= 6 + 1e-12))
