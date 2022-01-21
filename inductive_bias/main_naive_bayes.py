from inductive_bias.dataset import Dataset
from sklearn.naive_bayes import GaussianNB, CategoricalNB
import numpy as np
import utils
from sklearn.metrics import log_loss

from sklearn import preprocessing

NAIVE_BAYES = CategoricalNB()

if __name__ == '__main__':
    dataset = Dataset()
    train_features, train_labels, test_features, test_labels = dataset.load(
        'np', batch_size=None)

    train_features_raw = train_features.copy()

    if isinstance(NAIVE_BAYES, CategoricalNB):
        bins = preprocessing.KBinsDiscretizer(
            n_bins=8, encode='ordinal', strategy='uniform')
        bins.fit(train_features)
        train_features = bins.transform(train_features)
        test_features = bins.transform(test_features)
        filename = 'naive-bayes-categorical'
    else:
        filename = 'naive-bayes-gaussian'

    naives_bayes = NAIVE_BAYES
    naives_bayes.fit(train_features, train_labels)
    probs_test = naives_bayes.predict_proba(test_features)

    loss = log_loss(test_labels, probs_test)
    print(f'loss: {loss:.4f}')

    feature_0 = np.linspace(0.0, 0.291, 1000)
    feature_1 = np.linspace(0.07117, 0.2226, 1000)

    mesh = utils.create_mesh(feature_0, feature_1)

    if isinstance(NAIVE_BAYES, CategoricalNB):
        mesh = bins.transform(mesh)

    # Predict P(Y=1|X), discard P(Y=0|X)
    probs = naives_bayes.predict_proba(mesh)[:, 1]

    utils.plot_color_mesh(feature_0,
                          feature_1,
                          probs,
                          x_name='worst concave points',
                          y_name='worst smoothness',
                          inputs=train_features_raw,
                          labels=train_labels,
                          filename=f'inductive_bias/images/{filename}.png')
