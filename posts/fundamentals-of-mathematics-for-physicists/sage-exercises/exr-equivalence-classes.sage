# Exercise exr-equivalence-classes: the classes of 'same remainder modulo 3' on {0,...,11}, and the quotient set.
X = range(12)
classes = {}
for n in X:
    classes.setdefault(n % 3, []).append(n)
print("equivalence classes of n ~ m iff 3 | (n - m):", classes)
print("the quotient set has", len(classes), "elements, the remainders", sorted(classes), "; the classes partition X:", sorted(sum(classes.values(), [])) == list(X))
print("Sage's Zmod(3) is this quotient set with its addition:", Zmod(3).list())
