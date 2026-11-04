# Exercise exr-prob-bayes-sequential: Bayes' rule in odds form for a rare condition (prevalence 1/100, sensitivity
# 99/100, false-positive rate 1/20) and the posterior after one and after two conditionally independent positive tests.
prev, sens, fpr = QQ(1) / 100, QQ(99) / 100, QQ(1) / 20
prior_odds = prev / (1 - prev)
LR = sens / fpr
print("prior odds =", prior_odds, "; likelihood ratio of a positive test =", LR)
for tests in [1, 2, 3]:
    odds = prior_odds * LR ** tests
    post = odds / (1 + odds)
    print("after", tests, "positive test(s): posterior odds =", odds, ", posterior probability =", post, "=", numerical_approx(post, digits=4))
direct = prev * sens / (prev * sens + (1 - prev) * fpr)
print("direct Bayes after one test:", direct, "; agrees with the odds form:", direct == prior_odds * LR / (1 + prior_odds * LR))
