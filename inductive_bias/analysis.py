from inductive_bias.dataset import Dataset
import matplotlib.pyplot as plt
import scipy
import numpy as np


if __name__ == '__main__':
    dataset = Dataset()
    train_features, train_labels, test_features, test_labels = dataset.load(
        'np', batch_size=None)

    mean_0 = np.mean(train_features[train_labels == 0], axis=0)
    mean_1 = np.mean(train_features[train_labels == 1], axis=0)

    std_0 = np.std(train_features[train_labels == 0], axis=0)
    std_1 = np.std(train_features[train_labels == 1], axis=0)

    fig, ax = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(hspace=0.5)

    data = [
        [mean_0, std_0],  # label 0
        [mean_1, std_1]  # label 1
    ]
    for row_index, row in enumerate(data):
        mean = row[0]
        std = row[1]
        for col_index, col in enumerate(row):
            ax[row_index][col_index].set_title(
                f'{dataset.feature_names[col_index]} (Label={row_index})')
            x = np.linspace(mean[col_index]-0.2, mean[col_index]+0.2, 1000)
            ax[row_index][col_index].plot(
                x, scipy.stats.norm.pdf(x, mean[col_index], std[col_index]))
            ax[row_index][col_index].hist(
                train_features[:, 0], bins=8, density=True, histtype='stepfilled', alpha=0.5)

    plt.tight_layout()
    plt.savefig('inductive_bias/images/histograms.png', dpi=300)
    plt.close()
