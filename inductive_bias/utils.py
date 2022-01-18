import numpy as np
import matplotlib.pyplot as plt


def create_mesh(feature_0, feature_1):
    x, y = np.meshgrid(feature_0, feature_1)
    return np.vstack([x.ravel(), y.ravel()]).T


def plot_color_mesh(x_data, y_data, probs, inputs, labels, x_name, y_name, filename):
    probs = np.asarray(probs)
    probs = probs.reshape((x_data.shape[0], y_data.shape[0]))
    plt.pcolormesh(x_data, y_data, probs, shading='auto',
                   vmin=probs.min(), vmax=probs.max())
    plt.colorbar()

    if labels is not None and inputs is not None:
        plt.scatter(inputs[:, 0], inputs[:, 1],
                    c=labels, cmap=plt.cm.binary, s=1.0)

    plt.xlim(x_data.min(), x_data.max())
    plt.ylim(y_data.min(), y_data.max())
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
