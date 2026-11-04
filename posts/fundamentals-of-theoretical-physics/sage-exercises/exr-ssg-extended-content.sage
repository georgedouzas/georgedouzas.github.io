# Exercise exr-ssg-extended-content: the field content of the four-dimensional N-extended supergravity multiplets, read
# off from the CPT-completed helicity ladders with top helicity 2: a field of spin s >= 1/2 accounts for helicities +-s,
# a real scalar for one helicity-0 state. Compare with Tanii (hep-th/9802138), Table 3.
def single(N, top):
    lam0 = top - QQ(N) / 2
    return {lam0 + QQ(k) / 2: binomial(N, k) for k in range(N + 1)}
def cpt(N, top):
    s = single(N, top); out = dict(s)
    if min(s) != -top:
        for h, n in s.items(): out[-h] = out.get(-h, 0) + n
    return out
names = [(2, "graviton"), (QQ(3) / 2, "gravitini"), (1, "vectors"), (QQ(1) / 2, "spin-1/2"), (0, "real scalars")]
content = {}
for N in (1, 2, 3, 4, 5, 6, 8):
    st = cpt(N, 2)
    c = {name: (st.get(h, 0) if h == 0 else st.get(h, 0)) for h, name in names}
    content[N] = c
    print("N = %d:" % N, c)
expected = {1: (1, 1, 0, 0, 0), 2: (1, 2, 1, 0, 0), 3: (1, 3, 3, 1, 0), 4: (1, 4, 6, 4, 2), 5: (1, 5, 10, 11, 10), 6: (1, 6, 16, 26, 30), 8: (1, 8, 28, 56, 70)}
print("[PASS] the content agrees with Tanii's Table 3: N = 4 has 1, 4, 6, 4, 2 and N = 8 has 1, 8, 28, 56, 70:",
      all(tuple(content[N][nm] for _, nm in names) == expected[N] for N in expected))
print("[PASS] the helicity +s and -s multiplicities agree, so each count is a number of fields:", all(cpt(N, 2)[h] == cpt(N, 2)[-h] for N in expected for h in cpt(N, 2)))
