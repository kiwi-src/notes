from sklearn.naive_bayes import CategoricalNB
naives_bayes = CategoricalNB(alpha=0.0, fit_prior=True)

X_train = [[1], [1], [1], [0]]
y_train = [1, 0, 1, 0]

X_test = X_train
y_test = y_train

naives_bayes.fit(X_train, y_train)
probs = naives_bayes.predict_proba(X_test)

for x, p in zip(X_test, probs):
    print(f'P(Y=1|X={x[0]})={p[1]:.4f}')