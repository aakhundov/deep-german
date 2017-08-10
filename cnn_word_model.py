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


class CNNWordModel:
    """CNN-based TF model for inferring a class from character-level one-hot representation of a word."""

    def __init__(self, inputs, targets, dropout, filters,
                 window_size, num_layers, num_hidden, optimizer):

        self.inputs = inputs
        self.targets = targets
        self.dropout = dropout
        self.filters = filters
        self.window_size = window_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.optimizer = optimizer if optimizer \
            else tf.train.AdamOptimizer()

        self.input_dim = int(self.inputs.get_shape()[2])
        self.num_classes = int(self.targets.get_shape()[1])

        self.logits
        self.prediction
        self.error
        self.loss
        self.training

    @staticmethod
    def _conv_layer(x, in_channels, out_channels, window):
        weights = tf.Variable(tf.truncated_normal([window, in_channels, out_channels], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[out_channels]))

        return tf.nn.relu(tf.nn.conv1d(x, weights, stride=1, padding='SAME') + biases)

    @lazy_property
    def logits(self):
        conv1 = self._conv_layer(self.inputs, self.input_dim, self.filters[0], self.window_size)
        conv2 = self._conv_layer(conv1, self.filters[0], self.filters[1], self.window_size)
        last_layer = tf.reshape(conv2, [-1, self.input_dim * self.filters[1]])

        for i in range(self.num_layers):
            last_layer = layers.fully_connected(
                last_layer, self.num_hidden[i]
            )
            last_layer = tf.nn.dropout(
                last_layer, keep_prob=1.0-self.dropout
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
