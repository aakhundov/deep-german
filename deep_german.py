import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from datetime import datetime

from read_data import read_data_sets
from word_model import WordModel


MAX_WORD_LEN = 31
ALPHABET_SIZE = 31
NUM_GENDERS = 3

EPOCHS = 30

BATCH_SIZE = 128               # 128, 256, 512
NUM_LAYERS = 1                 # 1, 2, 3
CELL_TYPE = rnn.BasicRNNCell   # rnn.BasicRNNCell, rnn.BasicLSTMCell, rnn.GRUCell, rnn.LSTMCell(use_peepholes=True)
DROPOUT_RATE = 0.0             # 0.0, 0.5
LEARNING_RATE = 1e-3           # 1e-2, 1e-3, 1e-4

NUM_HIDDEN = [128, 128, 128]


def echo(*args):
    print("[{0}] ".format(datetime.now()), end="")
    print(*args)


def log(file, message=""):
    file.write(message + "\n")
    if message != "":
        echo(message)
    else:
        print()


model_name = "{0}_{1}_{2}_{3}_{4}".format(
    CELL_TYPE.__name__, NUM_LAYERS,
    LEARNING_RATE, DROPOUT_RATE, BATCH_SIZE
)

log_path = "./results/logs/" + model_name + ".txt"
model_path = "./results/models/" + model_name + ".ckpt"
log_file = open(log_path, "w")

print("cell type:", CELL_TYPE.__name__)
print("hidden layers:", NUM_LAYERS)
print("hidden units:", NUM_HIDDEN[:NUM_LAYERS])
print("learning rate:", LEARNING_RATE)
print("dropout rate:", DROPOUT_RATE)
print("batch size:", BATCH_SIZE)
print()


# BUILDING GRAPH

echo("Creating placeholders...")

xs = tf.placeholder(tf.float32, [None, MAX_WORD_LEN, ALPHABET_SIZE])
ys = tf.placeholder(tf.float32, [None, NUM_GENDERS])
seq = tf.placeholder(tf.int32, [None])
dropout = tf.placeholder(tf.float32)

echo("Creating model...")

model = WordModel(
    xs, ys, seq, dropout,
    CELL_TYPE, NUM_LAYERS, NUM_HIDDEN,
    tf.train.AdamOptimizer(LEARNING_RATE)
)


# PREPARING DATA

echo("Preparing data...")

# preparing words dataset
dataset = read_data_sets()

print()
echo("Training set:", dataset.train.words.shape[0])
echo("Validation set:", dataset.validation.words.shape[0])
echo("Testing set:", dataset.test.words.shape[0])
print()


# EXECUTING THE GRAPH

best_epoch = 0
best_val_error = 1.0
saver = tf.train.Saver()

with tf.Session() as sess:
    echo("Initializing variables...")

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    echo("Training...")
    print()

    steps_per_epoch = dataset.train.words.shape[0] // BATCH_SIZE

    for epoch in range(1, EPOCHS+1):
        for step in range(steps_per_epoch):
            batch_xs, batch_ys, seq_len = dataset.train.next_batch(BATCH_SIZE)

            sess.run(
                model.training,
                feed_dict={
                    xs: batch_xs,
                    ys: batch_ys,
                    seq: seq_len,
                    dropout: DROPOUT_RATE
                }
            )

        val_loss, val_error = sess.run(
            [model.loss, model.error],
            feed_dict={
                xs: dataset.validation.words,
                ys: dataset.validation.genders,
                seq: dataset.validation.seq_length,
                dropout: 0.0
            }
        )

        if val_error < best_val_error:
            best_epoch = epoch
            best_val_error = val_error
            saver.save(sess, model_path)

        log(log_file, "Epoch {:2d}:  error {:3.2f}%  loss {:.4f}".format(
            epoch, 100 * val_error, val_loss
        ))

    saver.restore(sess, model_path)

    test_loss, test_error = sess.run(
        [model.loss, model.error],
        feed_dict={
            xs: dataset.test.words,
            ys: dataset.test.genders,
            seq: dataset.test.seq_length,
            dropout: 0.0
        }
    )

    log(log_file)
    log(log_file, "Best epoch: {0}".format(best_epoch))
    log(log_file)
    log(log_file, "Test Set:  error {:3.2f}%  loss {:.4f}".format(
        100 * test_error, test_loss
    ))

log_file.close()
