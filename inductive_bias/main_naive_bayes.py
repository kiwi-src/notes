from inductive_bias.dataset import Dataset
from sklearn.naive_bayes import GaussianNB
import numpy as np
import utils
from sklearn.metrics import log_loss

if __name__ == '__main__':
    dataset = Dataset()
    train_inputs, train_labels, test_inputs, test_labels = dataset.load(
        'np', batch_size=None)

    naives_bayes = GaussianNB()
    naives_bayes.fit(train_inputs, train_labels)
    probs_test = naives_bayes.predict_proba(test_inputs)

    loss = log_loss(test_labels, probs_test)
    print(f'loss: {loss:.4f}')

    feature_0 = np.linspace(0.0, 0.291, 1000)
    feature_1 = np.linspace(0.07117, 0.2226, 1000)
    mesh = utils.create_mesh(feature_0, feature_1)

    # Predict P(Y=1|X), discard P(Y=0|X)
    probs = naives_bayes.predict_proba(mesh)[:, 1]

    utils.plot_color_mesh(feature_0,
                          feature_1,
                          probs,
                          x_name='worst concave points',
                          y_name='worst smoothness',
                          inputs=train_inputs,
                          labels=train_labels,
                          filename='inductive_bias/images/naive-bayes.png')
