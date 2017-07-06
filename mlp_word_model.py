import functools

import tensorflow as tf
import tensorflow.contrib.layers as layers


def lazy_property(func):
    attribute = '_cache_' + func.__name__

    @property
    @functools.wraps(func)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.name_scope(func.__name__):
                setattr(self, attribute, func(self))
        return getattr(self, attribute)

    return decorator


class MLPWordModel:
    """MLP-based TF model for inferring a class from character-level one-hot representation of a word."""

    def __init__(self, inputs, targets, dropout, activation,
                 num_layers, num_hidden, optimizer):

        self.inputs = inputs
        self.targets = targets
        self.dropout = dropout
        self.activation = activation
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.optimizer = optimizer if optimizer \
            else tf.train.AdamOptimizer()

        self.num_classes = int(self.targets.get_shape()[1])

        self.logits
        self.prediction
        self.error
        self.loss
        self.training

    @lazy_property
    def logits(self):
        last_layer = self.inputs
        for i in range(self.num_layers):
            last_layer = layers.fully_connected(
                last_layer, self.num_hidden[i],
                activation_fn=self.activation
            )
            last_layer = tf.nn.dropout(
                last_layer, keep_prob=1-self.dropout
            )

        return layers.fully_connected(
            last_layer, num_outputs=self.num_classes,
            activation_fn=None
        )

    @lazy_property
    def prediction(self):
        return tf.nn.softmax(self.logits)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.targets, 1),
            tf.argmax(self.logits, 1)
        )

        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @lazy_property
    def loss(self):
        return tf.losses.softmax_cross_entropy(
            self.targets, self.logits)

    @lazy_property
    def training(self):
        return self.optimizer.minimize(self.loss)
