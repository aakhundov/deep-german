import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data


# PARAMETERS

DATA_FOLDER = "MNIST_data/"

EPOCHS = 10
BATCH_SIZE = 100

IMG_ROWS = 28
IMG_COLS = 28
IMG_LABELS = 10

TIME_STEPS = 28
TIME_OFFSET = 0
NUM_LAYERS = 2
NUM_HIDDEN = [256, 128]
CELL_TYPE = rnn.GRUCell
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001


def echo(*message):
    print("[{0}] ".format(datetime.now().time()), end="")
    print(*message)


def get_last_output_from_static_rnn(input_images, rnn_cell):
    """Create RNN by means of tf.static_rnn() and return last output"""
    image_slice = input_images[:, TIME_OFFSET: TIME_OFFSET + TIME_STEPS, :]

    rnn_outputs, rnn_state = rnn.static_rnn(
        rnn_cell, tf.unstack(image_slice, axis=1), dtype=tf.float32
    )

    last = rnn_outputs[-1]

    return last


def get_last_output_from_dynamic_rnn(input_images, rnn_cell):
    """Create RNN by means of tf.dynamic_rnn() and return last output"""
    image_slice = input_images[:, TIME_OFFSET: TIME_OFFSET + TIME_STEPS, :]

    rnn_outputs, rnn_state = tf.nn.dynamic_rnn(
        rnn_cell, image_slice, dtype=tf.float32
    )

    trans_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
    last_partition = tf.dynamic_partition(
        trans_outputs, [0] * (TIME_STEPS - 1) + [1], 2
    )[1]

    return tf.reshape(last_partition, [-1, NUM_HIDDEN])


# BUILDING GRAPH

echo("Creating placeholders...")

# setting up placeholders
xs = tf.placeholder(tf.float32, [None, IMG_ROWS * IMG_COLS])
ys = tf.placeholder(tf.float32, [None, IMG_LABELS])
dropout = tf.placeholder(tf.float32)

echo("Creating cells...")

# setting up RNN cells
layer_cells = []
for i in range(NUM_LAYERS):
    layer_cell = CELL_TYPE(NUM_HIDDEN[i])
    if DROPOUT_RATE < 1.0:
        layer_cell = rnn.DropoutWrapper(layer_cell, output_keep_prob=1.0-dropout)
    layer_cells.append(layer_cell)
if len(layer_cells) > 1:
    cell = rnn.MultiRNNCell(layer_cells)
else:
    cell = layer_cells[0]

# reshaping MNIST data in a square form
images = tf.reshape(xs, [-1, IMG_ROWS, IMG_COLS])

echo("Creating network...")

# fetching last output of the RNN
last_output = get_last_output_from_static_rnn(images, cell)

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

echo("Computing gradients...")

# minimization
train = optimizer.minimize(loss)


# PREPARING DATA

echo("Preparing data...")

# (download and) extract MNIST dataset
print()
dataset = input_data.read_data_sets(DATA_FOLDER, one_hot=True)
print()
echo("Training set:", dataset.train.images.shape[0])
echo("Validation set:", dataset.validation.images.shape[0])
echo("Testing set:", dataset.test.images.shape[0])
print()


# EXECUTING THE GRAPH

with tf.Session() as sess:
    echo("Initializing variables...")

    sess.run(tf.global_variables_initializer())

    echo("Training...")
    print()

    steps_per_epoch = dataset.train.images.shape[0] // BATCH_SIZE

    for epoch in range(EPOCHS):
        for step in range(steps_per_epoch):
            batch_xs, batch_ys = dataset.train.next_batch(BATCH_SIZE)

            sess.run(train, feed_dict={
                xs: batch_xs,
                ys: batch_ys,
                dropout: DROPOUT_RATE
            })

        val_error = sess.run(
            error,
            feed_dict={
                xs: dataset.validation.images,
                ys: dataset.validation.labels,
                dropout: 0.0
            }
        )

        echo('Epoch {:2d}  error {:3.2f}%'.format(
            epoch + 1,
            100 * val_error
        ))

    test_error = sess.run(
        error,
        feed_dict={
            xs: dataset.test.images,
            ys: dataset.test.labels,
            dropout: 0.0
        }
    )

    print()
    echo("Test error {:3.2f}% ".format(
        100 * test_error
    ))
