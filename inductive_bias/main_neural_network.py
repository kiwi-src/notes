import tensorflow as tf
from trainer_config import TrainerConfig
import time
from dataset import Dataset
import numpy as np
import utils

tf.random.set_seed(1337)

class Model(tf.keras.Model):
    def __init__(self, config, num_classes):
        super(Model, self).__init__()
        self._config = config

        if config.batch_norm:
            self.batch_norm_layer = tf.keras.layers.BatchNormalization()
            self.batch_norm_layer2 = tf.keras.layers.BatchNormalization()
            self.batch_norm_layer3 = tf.keras.layers.BatchNormalization()

        self.layer_1 = tf.keras.layers.Dense(units=config.num_units,
                                             kernel_regularizer=tf.keras.regularizers.l2(
                                                 config.l2),
                                             activation=tf.keras.activations.gelu)

        if config.hidden:
            self.layer_2 = tf.keras.layers.Dense(units=16,
                                                 kernel_regularizer=tf.keras.regularizers.l2(
                                                     config.l2),
                                                 activation=tf.keras.activations.gelu)
        self.layer_3 = tf.keras.layers.Dense(
            units=num_classes, activation=None)

    def call(self, features):
        if self._config.batch_norm:
            layer = self.layer_1(self.batch_norm_layer(features))
        else:
            layer = self.layer_1(features)

        if self._config.batch_norm:
            layer = self.batch_norm_layer2(layer)

        if self._config.hidden:
            layer = self.layer_2(layer)
            if self._config.batch_norm:
                layer = self.batch_norm_layer3(layer)

        return self.layer_3(layer)

    def compute_regularizer_loss(self):
        loss = 0
        for layer in self.layers:
            if hasattr(layer, 'layers') and layer.layers:
                raise NotImplementedError
            if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer:
                loss += layer.kernel_regularizer(layer.kernel)
            if hasattr(layer, 'bias_regularizer') and layer.bias_regularizer:
                loss += layer.bias_regularizer(layer.bias)
        return loss

if __name__ == '__main__':
    val_auc = tf.keras.metrics.AUC(name='val_auc', num_thresholds=200)

    # Compute mean cross entropy loss on train set for one epoch
    train_ce_loss = tf.keras.metrics.Mean(name='train_ce_loss')

    # Compute mean cross entropy loss on eval set for one epoch
    val_ce_loss = tf.keras.metrics.Mean(name='val_ce_loss')

    NUM_EPOCHS = 1000
    config = TrainerConfig(
        learning_rate=1e-3,
        regularization=False,
        l2=0.001,
        num_units=256,
        optimizer=tf.keras.optimizers.Adam,
        batch_norm=True,
        batch_size=None,
        num_examples=None,
        hidden=True
    )

    dataset = Dataset()
    cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    optimizer = config.optimizer(learning_rate=config.learning_rate)
    model = Model(config, num_classes=2)

    @tf.function()
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            mean_cross_entropy_loss = cross_entropy_loss(
                tf.expand_dims(labels, axis=1), logits)
            mean_loss = mean_cross_entropy_loss + model.compute_regularizer_loss()

        if config.regularization:
            gradients = tape.gradient(mean_loss, model.trainable_variables)
        else:
            gradients = tape.gradient(
                mean_cross_entropy_loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_ce_loss(mean_cross_entropy_loss)

    @tf.function
    def val_step(inputs, labels):
        logits = model(inputs, training=False)
        val_auc(y_true=tf.one_hot(labels, depth=2),
                 y_pred=tf.nn.softmax(logits))
        mean_cross_entropy_loss = cross_entropy_loss(
            tf.expand_dims(labels, axis=1), logits)
        val_ce_loss(mean_cross_entropy_loss)

    train_examples, eval_examples = dataset.load(
        'tf', batch_size=config.batch_size, num_examples=config.num_examples)
    start = time.time()

    best_eval_auc = 0.0
    best_eval_loss = float('inf')
    best_train_loss = float('inf')
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        val_auc.reset_states()
        train_ce_loss.reset_states()
        val_ce_loss.reset_states()

        # training
        for input, label in train_examples:
            train_step(input, label)

        # validation
        for input, label in eval_examples:
            val_step(input, label)

        log = f"Epoch {epoch} " \
            f"- {train_ce_loss.name} {train_ce_loss.result():.6f} " \
            f"- {val_ce_loss.name} {val_ce_loss.result():.6f} " \
            f"- {val_auc.name} {val_auc.result():.6f} "
        print(log)
        if val_ce_loss.result() < best_eval_loss:
            best_train_loss = train_ce_loss.result()
            best_eval_loss = val_ce_loss.result()
            best_eval_auc = val_auc.result()
            best_epoch = epoch
            best_log = log
            best_model = model

        print("BEST",f"{best_log}")
    
    inputs = []
    labels = []
    for input, label in train_examples:
        inputs.append(input)
        labels.append(label)

    inputs = tf.concat(inputs, axis=0)
    labels = tf.concat(labels, axis=0)

    max_0 = tf.math.reduce_max(inputs[:,0], axis=0)
    min_0 = tf.math.reduce_min(inputs[:,0], axis=0)
    max_1 = tf.math.reduce_max(inputs[:,1], axis=0)
    min_1 = tf.math.reduce_min(inputs[:,1], axis=0)
    feature_0 = np.linspace(min_0, max_0, 1000)
    feature_1 = np.linspace(min_1, max_1, 1000)

    mesh = utils.create_mesh(feature_0=feature_0, feature_1=feature_1)    
    logits = best_model(mesh, training=False)
    probs = tf.nn.softmax(logits)[:,1]

    utils.plot_color_mesh(feature_0, 
                          feature_1, 
                          probs, 
                          inputs, 
                          labels, 
                          x_name='worst concave points', 
                          y_name='worst smoothness', 
                          filename='inductive_bias/images/neural-network.png')
