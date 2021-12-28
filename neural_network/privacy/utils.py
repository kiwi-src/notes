from jax import random
import jax.numpy as jnp
import jax
import csv
import os

BASE_DIR = os.path.join("neural_network", "privacy")

def write(rows):
    with open(os.path.join(BASE_DIR, "points.csv"), "w") as file:
        writer = csv.writer(file)
        writer.writerows(rows)

def read():
    with open(os.path.join(BASE_DIR, "points.csv"), "r") as file:
        reader = csv.reader(file)
        return list(reader)

def random_params(NUM_FEATURES, BATCH_SIZE):
    PRNG_KEY = random.PRNGKey(2)
    weights_key, bias_key, inputs_fake_key = random.split(PRNG_KEY, 3)
    weights = random.normal(weights_key, (NUM_FEATURES,))
    bias = random.normal(bias_key, ())
    inputs_fake = random.normal(inputs_fake_key, (BATCH_SIZE, 1))
    return weights, bias, inputs_fake

def predict(weights, bias, inputs):
    logits = jnp.dot(inputs, weights) + bias
    return jax.nn.sigmoid(logits)

def binary_cross_entropy(weights, bias, inputs, labels):
    probs = predict(weights, bias, inputs)
    loss = (labels * jnp.log(probs) + (1.0 - labels) * jnp.log(1.0 - probs))
    return -jnp.mean(loss)

def dataset():
    # We assume a batch size of 2
    inputs = jnp.array([[0.2], [0.9]])

    # We assume that the labels are known
    labels = jnp.array([1, 1])
    return inputs, labels