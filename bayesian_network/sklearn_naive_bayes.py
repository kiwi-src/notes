from pyexpat import features
from sklearn.naive_bayes import CategoricalNB

def one_feature():
    naives_bayes = CategoricalNB(alpha=0.0, fit_prior=True)
    data = [[1], [1], [1], [0]]
    label = [1, 0, 1, 0]

    naives_bayes.fit(data, label)
    probs = naives_bayes.predict_proba(data)

    for features, p in zip(data, probs):
        print(f'P(Y=1|X={features[0]})={p[1]:.4f}')

def two_features():
    naives_bayes = CategoricalNB(alpha=0.0, fit_prior=True)
    data = [[0, 1], [1, 1], [0, 0], [1,1]]
    label = [1, 1, 1, 0]

    naives_bayes.fit(data, label)
    probs = naives_bayes.predict_proba(data)

    for features, p in zip(data, probs):
        print(f'P(Y=1|X1={features[0]},X2={features[1]})={p[1]:.4f}')


if __name__ == '__main__':
    #one_feature()
    two_features()