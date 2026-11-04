# Exercise exr-prob-monty-hall: exact enumeration of the Monty Hall game. Outcomes are (prize, choice, opened door)
# with the host opening a prize-free unchosen door, uniformly when there are two; switching wins with probability 2/3.
outcomes = []   # (prize, choice, opened, probability)
for prize in [1, 2, 3]:
    for choice in [1, 2, 3]:
        options = [d for d in [1, 2, 3] if d != prize and d != choice]
        for opened in options:
            outcomes.append((prize, choice, opened, QQ(1) / 9 / len(options)))
print("number of weighted outcomes:", len(outcomes), "; total probability:", sum(o[3] for o in outcomes))
stay = sum(o[3] for o in outcomes if o[1] == o[0])
switch = sum(o[3] for o in outcomes if o[1] != o[0])
print("P(stay wins) =", stay, "; P(switch wins) =", switch)
# conditional on choice 1 and host opening door 3
given = [o for o in outcomes if o[1] == 1 and o[2] == 3]
pg = sum(o[3] for o in given)
print("given choice 1 and door 3 opened: P(prize behind 2) =", sum(o[3] for o in given if o[0] == 2) / pg, ", P(prize behind 1) =", sum(o[3] for o in given if o[0] == 1) / pg)
