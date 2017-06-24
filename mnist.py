import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from tensorflow.examples.tutorials.mnist import input_data


# PARAMETERS

DATA_FOLDER = "MNIST_data/"

EPOCHS = 10
TRAIN_STEPS = 550
BATCH_SIZE = 100

IMG_ROWS = 28
IMG_COLS = 28
IMG_LABELS = 10

TIME_STEPS = 28
TIME_OFFSET = 0
NUM_HIDDEN = 128
LEARNING_RATE = 0.001


# BUILDING GRAPH

# setting up RNN cells
cell = rnn.BasicLSTMCell(NUM_HIDDEN)

# setting up placeholders
xs = tf.placeholder(tf.float32, [None, IMG_ROWS * IMG_COLS])
ys = tf.placeholder(tf.float32, [None, IMG_LABELS])

# reshaping MNIST data in a square form
images = tf.reshape(xs, [-1, IMG_ROWS, IMG_COLS])


def create_rnn_with_static_rnn():
    """Create RNN by means of tf.static_rnn() and return last output"""
    image_slice = images[:, TIME_OFFSET: TIME_OFFSET + TIME_STEPS, :]

    rnn_outputs, rnn_state = rnn.static_rnn(
        cell, tf.unstack(image_slice, axis=1), dtype=tf.float32
    )

    last = rnn_outputs[-1]

    return last


def create_rnn_with_dynamic_rnn():
    """Create RNN by means of tf.dynamic_rnn() and return last output"""
    image_slice = images[:, TIME_OFFSET: TIME_OFFSET + TIME_STEPS, :]

    rnn_outputs, rnn_state = tf.nn.dynamic_rnn(
        cell, image_slice, dtype=tf.float32
    )

    trans_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
    last_partition = tf.dynamic_partition(
        trans_outputs, [0] * (TIME_STEPS - 1) + [1], 2
    )[1]

    return tf.reshape(last_partition, [-1, NUM_HIDDEN])


# fetching last output of the RNN
print("Creating network...")
last_output = create_rnn_with_dynamic_rnn()

# inference artifacts
logits = tf.contrib.layers.fully_connected(
    last_output, IMG_LABELS, activation_fn=None
)

# evaluation artifacts
mistakes = tf.not_equal(tf.argmax(ys, 1), tf.argmax(logits, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

# training artifacts
loss = tf.losses.softmax_cross_entropy(ys, logits)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
print("Computing gradients...")
train = optimizer.minimize(loss)


# PREPARING DATA

# (download and) extract MNIST dataset
print("Preparing data...")
print()
dataset = input_data.read_data_sets(DATA_FOLDER, one_hot=True)
print()
print("Training set:", dataset.train.images.shape[0])
print("Validation set:", dataset.validation.images.shape[0])
print("Testing set:", dataset.test.images.shape[0])
print()


# EXECUTING GRAPH

with tf.Session() as sess:
    print("Initializing variables...")
    sess.run(tf.global_variables_initializer())

    print("Training...")
    print()

    for epoch in range(EPOCHS):
        for step in range(TRAIN_STEPS):
            batch_xs, batch_ys = dataset.train.next_batch(BATCH_SIZE)

            sess.run(train, feed_dict={
                xs: batch_xs,
                ys: batch_ys
            })

        val_error = sess.run(
            error,
            feed_dict={
                xs: dataset.validation.images,
                ys: dataset.validation.labels
            }
        )

        print('Epoch {:2d}  error {:3.2f}%'.format(
            epoch + 1,
            100 * val_error
        ))

    test_error = sess.run(
        error,
        feed_dict={
            xs: dataset.test.images,
            ys: dataset.test.labels
        }
    )

    print()
    print("Test error {:3.2f}% ".format(
        100 * test_error
    ))
