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
        bins_feature_0 = preprocessing.KBinsDiscretizer(
            n_bins=10, encode='ordinal', strategy='uniform')
        bins_feature_1 = preprocessing.KBinsDiscretizer(
            n_bins=18, encode='ordinal', strategy='uniform')

        train_feature_0 = np.expand_dims(train_features[:,0],axis=1)
        train_feature_1 = np.expand_dims(train_features[:,1],axis=1)
        test_feature_0 = np.expand_dims(test_features[:,0],axis=1)
        test_feature_1 = np.expand_dims(test_features[:,1],axis=1)
    
        # Fit discretizer values with train data
        bins_feature_0.fit(train_feature_0)
        bins_feature_1.fit(train_feature_1)

        # Transform feature values
        train_feature_0 = bins_feature_0.transform(train_feature_0)
        train_feature_1 = bins_feature_1.transform(train_feature_1)
        test_feature_0 = bins_feature_0.transform(test_feature_0)
        test_feature_1 = bins_feature_1.transform(test_feature_1)

        train_features = np.concatenate((train_feature_0,train_feature_1),axis=1)
        test_features = np.concatenate((test_feature_0,test_feature_1),axis=1)

        filename = 'naive-bayes-categorical'
    else:
        filename = 'naive-bayes-gaussian'

    naives_bayes = NAIVE_BAYES
    naives_bayes.fit(train_features, train_labels)
    probs_test = naives_bayes.predict_proba(test_features)

    loss = log_loss(test_labels, probs_test)
    print(f'loss: {loss:.4f}')

    feature_0_raw = np.linspace(0.0, 0.291, 1000)
    feature_1_raw = np.linspace(0.07117, 0.2226, 1000)

    if isinstance(NAIVE_BAYES, CategoricalNB):
        feature_0 = np.expand_dims(feature_0_raw, axis=1)
        feature_0 = bins_feature_0.transform(feature_0)
        feature_1 = np.expand_dims(feature_1_raw, axis=1)
        feature_1 = bins_feature_1.transform(feature_1)

    mesh = utils.create_mesh(feature_0, feature_1)

    # Predict P(Y=1|X), discard P(Y=0|X)
    probs = naives_bayes.predict_proba(mesh)[:, 1]

    utils.plot_color_mesh(feature_0_raw,
                          feature_1_raw,
                          probs,
                          x_name='worst concave points',
                          y_name='worst smoothness',
                          inputs=train_features_raw,
                          labels=train_labels,
                          filename=f'inductive_bias/images/{filename}.png')
