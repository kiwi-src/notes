import jax.numpy as jnp
from jax import grad
import jax
import utils


def predict(weights, bias, inputs):
    logits = jnp.dot(inputs, weights) + bias
    return jax.nn.sigmoid(logits)


def regression_diff_gradients(weights, bias, inputs, labels,
                              gradients_true):
    # inputs.shape = (batch_size)
    # gradients_true.shape = (batch_size)
    gradients_fake = grad(utils.binary_cross_entropy, argnums=(0, 1))(
        weights, bias, inputs, labels)
    diff_weights = jnp.square(gradients_true[0] - gradients_fake[0])
    diff_bias = jnp.square(gradients_true[1] - gradients_fake[1])
    loss = diff_weights + diff_bias
    return jnp.mean(loss)


def main():
    BATCH_SIZE = 2
    NUM_FEATURES = 1
    LEARNING_RATE = 1.0
    NUM_TRAIN_STEPS = 6001
    weights, bias, inputs_fake = utils.random_params(NUM_FEATURES, BATCH_SIZE)
    inputs_true, labels_true = utils.dataset()

    # Execute one train step in order to compute the true gradients
    gradients_true = grad(utils.binary_cross_entropy, argnums=(0, 1))(
        weights, bias, inputs_true, labels_true)

    points = []
    for i in range(1, NUM_TRAIN_STEPS):
        # Compute gradient regression_diff_gradients wrt inputs_fake
        gradients_inputs_fake = grad(regression_diff_gradients, argnums=(2))(
            weights, bias, inputs_fake, labels_true, gradients_true)
        loss = regression_diff_gradients(
            weights, bias, inputs_fake, labels_true, gradients_true)
        print(f'step {i} – loss {loss}')
        print(f"inputs_fake[0]: {inputs_fake[0][0]}")
        print(f"inputs_fake[1]: {inputs_fake[1][0]}")
        points.append([i, inputs_fake[0][0], inputs_fake[1][0]])

        # Update fake inputs
        inputs_fake = inputs_fake - LEARNING_RATE * gradients_inputs_fake

    # Save path to csv file
    utils.write(points)

    print("\nRESULTS")
    # gradient binary_cross_entropy wrt inputs_true
    gradients_bce_inputs_true = grad(utils.binary_cross_entropy, argnums=(0, 1))(
        weights, bias, inputs_true, labels_true)
    print(f'gradients_bce_inputs_true: {gradients_bce_inputs_true}')

    # gradient binary_cross_entropy wrt inputs_fake
    gradients_bce_inputs_fake = grad(utils.binary_cross_entropy, argnums=(0, 1))(
        weights, bias, inputs_fake, labels_true)
    print(f'gradients_bce_inputs_fake: {gradients_bce_inputs_fake}')

    print("\n")
    print(f'inputs_true: {["%.2f"%x for x in inputs_true.flatten().tolist()]}')
    print(f'inputs_fake: {["%.2f"%x for x in inputs_fake.flatten().tolist()]}')


if __name__ == '__main__':
    main()
