import jax.numpy as jnp
from jax import grad
import utils
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    NUM_FEATURES = 1
    NUM_POINTS = 41
    inputs, labels = utils.dataset()
    weights, bias, _ = utils.random_params(NUM_FEATURES, 2)
    gradients_inputs_true = grad(utils.binary_cross_entropy, argnums=(0, 1))(
        weights, bias, inputs, labels)

    x1 = jnp.linspace(0, 1, NUM_POINTS)
    x2 = jnp.linspace(0, 1, NUM_POINTS)

    gradient_diffs = []
    best_diff = float('inf')
    grid = jnp.meshgrid(x1, x2)
    combinations = jnp.array(grid).T.reshape(-1, 2)

    gradient_diffs = []
    for x in combinations:
        inputs_pred = jnp.array([[x[0]], [x[1]]])
        gradients_inputs_fake = grad(utils.binary_cross_entropy, argnums=(0, 1))(
            weights, bias, inputs_pred, labels)
        gradient_diff = (
            gradients_inputs_fake[0][0] - gradients_inputs_true[0][0]) ** 2
        gradient_diffs.append(gradient_diff)
        if gradient_diff < best_diff:
            best_diff = gradient_diff
            print(f'x0 {x[0]:.6f} – x1 {x[1]:.6f} – best_diff {best_diff:.6f}')

    # Plot gradient diffs
    gradient_diffs = np.array(gradient_diffs)
    gradient_diffs = gradient_diffs.reshape((NUM_POINTS, NUM_POINTS))
    plot = plt.pcolormesh(grid[1], grid[0], gradient_diffs, cmap='Blues',
                          vmin=gradient_diffs.min(), vmax=gradient_diffs.max())

    # Plot true input
    plt.scatter(inputs[0], inputs[1], c='r', s=10.0)

    # Plot points
    points = utils.read()
    points = [[int(i), float(x1), float(x2)] for i, x1, x2 in points]
    for _, x1, x2 in points:
        plt.scatter(x1, x2, c='black', s=1.0)

    plt.annotate(f"Step {points[-1][0]}", (points[-1][1]+0.005, points[-1][2]),
                 arrowprops=dict(arrowstyle="->", color='black'), xytext=(points[-1][1]+0.1, points[-1][2]-0.013))

    plt.annotate(f"Step {points[99][0]}", (points[99][1]+0.005, points[99][2]),
                 arrowprops=dict(arrowstyle="->", color='black'), xytext=(points[99][1]+0.1, points[99][2]-0.013))

    plt.xlabel(r"${x_1}$")
    plt.ylabel(r"${x_2}$")
    plt.colorbar(plot)
    plt.savefig(os.path.join(utils.BASE_DIR, "loss.png"), dpi=300)


if __name__ == '__main__':
    main()
