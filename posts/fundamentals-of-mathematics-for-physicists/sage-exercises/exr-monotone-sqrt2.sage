# Exercise exr-monotone-sqrt2: the recursion a_{n+1} = (a_n + 2/a_n)/2 with a_0 = 2 is decreasing and bounded below
# by sqrt(2), so it converges by the monotone convergence theorem; the limit L satisfies L = (L + 2/L)/2, i.e. L^2 = 2.
a = [QQ(2)]
for _ in range(6):
    a.append((a[-1] + 2 / a[-1]) / 2)
print("terms:", [float(t) for t in a])
print("decreasing:", all(a[i + 1] < a[i] for i in range(len(a) - 1)), "; bounded below by sqrt(2):", all(t ** 2 > 2 for t in a))
print("|a_6 - sqrt 2| =", float(abs(a[6] - sqrt(2))), "(quadratic convergence: the number of correct digits doubles each step)")
L = var("L")
print("fixed points of L = (L + 2/L)/2:", solve(L == (L + 2 / L) / 2, L))
