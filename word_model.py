import functools

import tensorflow as tf
import tensorflow.contrib.rnn as rnn


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


class WordModel:
    """TensorFlow model for inferring a class from character-level one-hot representation of a word."""

    def __init__(self, inputs, targets, seq_length, dropout,
                 cell_type, num_layers, num_hidden, optimizer):

        self.inputs = inputs
        self.targets = targets
        self.seq_length = seq_length
        self.dropout = dropout
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.optimizer = optimizer if optimizer \
            else tf.train.AdamOptimizer()

        self.batch_size = tf.shape(self.inputs)[0]
        self.num_classes = int(self.targets.get_shape()[1])

        self.cell
        self.last_output
        self.logits
        self.prediction
        self.error
        self.loss
        self.training

    @lazy_property
    def cell(self):
        layers = []
        for i in range(self.num_layers):
            layer = self.cell_type(self.num_hidden[i])
            layer = rnn.DropoutWrapper(layer, output_keep_prob=1.0 - self.dropout)
            layers.append(layer)
        if self.num_layers > 1:
            return rnn.MultiRNNCell(layers)
        else:
            return layers[0]

    @lazy_property
    def last_output(self):
        rnn_outputs, rnn_state = tf.nn.dynamic_rnn(
            self.cell, self.inputs,
            sequence_length=self.seq_length,
            dtype=self.inputs.dtype
        )

        indices = tf.stack([
            tf.range(0, self.batch_size),
            self.seq_length - 1
        ], axis=1)

        return tf.gather_nd(rnn_outputs, indices)

    @lazy_property
    def logits(self):
        return tf.contrib.layers.fully_connected(
            self.last_output, self.num_classes,
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
