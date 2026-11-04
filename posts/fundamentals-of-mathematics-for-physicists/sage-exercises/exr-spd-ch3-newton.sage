# Exercise exr-spd-ch3-newton: ch_3 = (c1^3 - 3 c1 c2 + 3 c3)/6 in terms of Chern classes (formal roots y1, y2, y3).
y1, y2, y3 = var("y1 y2 y3")
c1 = y1 + y2 + y3; c2 = y1 * y2 + y1 * y3 + y2 * y3; c3 = y1 * y2 * y3
ch3 = (y1 ^ 3 + y2 ^ 3 + y3 ^ 3) / 6
print("ch_3 - (c1^3 - 3 c1 c2 + 3 c3)/6 =", (ch3 - (c1 ^ 3 - 3 * c1 * c2 + 3 * c3) / 6).expand())
print("with c1 = 0: ch_3 = c3/2, so int ch_3 = chi/2; the term -(p1/24) ch_1 = -(p1/24) c1 vanishes")
