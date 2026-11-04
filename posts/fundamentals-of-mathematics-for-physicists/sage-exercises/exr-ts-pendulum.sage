# Exercise exr-ts-pendulum (Foulaadvand, Sec. 3.7, eqs. (3.45)-(3.53), Fig. 3.20): the period of the pendulum
# theta'' = -w0^2 sin(theta) with w0 = 3, released from rest at theta0, from the Euler-Cromer and velocity Verlet
# steps (tau = 0.001, period from the first two downward zero crossings, interpolated), against the exact
# 4K(sin^2(theta0/2))/w0 and the series 2 pi/w0 (1 + theta0^2/16 + 11 theta0^4/3072).
w0 = 3.0; tau = 0.001
def period(theta0, method):
    th, om, t = theta0, 0.0, 0.0
    crossings = []
    while len(crossings) < 2:
        if method == "EC":
            om_new = om - w0^2*sin(th)*tau; th_new = th + om_new*tau
        else:
            a = -w0^2*sin(th); th_new = th + om*tau + a*tau^2/2; om_new = om + (a - w0^2*sin(th_new))*tau/2
        if th > 0 and th_new <= 0:
            crossings.append(t + tau*th/(th - th_new))
        th, om, t = th_new, om_new, t + tau
    return crossings[1] - crossings[0]
print(" theta0    exact    series   Euler-Cromer  Verlet")
for theta0 in [0.1, 0.2, 0.4, 0.8, 1.5, 3.0]:
    exact = (4*elliptic_kc(sin(theta0/2)^2)/w0).n()
    series = (2*pi/w0*(1 + theta0^2/16 + 11*theta0^4/3072)).n()
    print("%6.2f  %8.5f  %8.5f  %8.5f     %8.5f" % (theta0, exact, series, period(theta0, "EC"), period(theta0, "V")))
print("small-angle period 2 pi/w0 = %.5f" % (2*pi/w0).n())
