import sys
import tensorflow as tf

from datetime import datetime

from read_data import read_data_sets
from cnn_word_model import CNNWordModel


MAX_WORD_LEN = 31
ALPHABET_SIZE = 31
NUM_GENDERS = 3

EPOCHS = 50

BATCH_SIZE = 128           # -batch 128, 256, or 512
NUM_LAYERS = 3             # -layers 0, 1, 2, or 3
DROPOUT_RATE = 0.5         # -dropout 0.0 or 0.5
LEARNING_RATE = 1e-3       # -learning 1e-2, 1e-3, or 1e-4
WINDOW_SIZE = 7            # -window 3, 5, 7, or 9

CONV_FILTERS = [32, 64]
NUM_HIDDEN = [512, 256, 128]


def echo(*args):
    print("[{0}] ".format(datetime.now()), end="")
    print(*args)


def log(file, message=""):
    file.write(message + "\n")
    if message != "":
        echo(message)
    else:
        print()


while len(sys.argv) > 1:
    option = sys.argv[1]; del sys.argv[1]

    if option == "-batch":
        BATCH_SIZE = int(sys.argv[1]); del sys.argv[1]
    elif option == "-window":
        WINDOW_SIZE = int(sys.argv[1]); del sys.argv[1]
    elif option == "-dropout":
        DROPOUT_RATE = float(sys.argv[1]); del sys.argv[1]
    elif option == "-learning":
        LEARNING_RATE = float(sys.argv[1]); del sys.argv[1]
    else:
        print(sys.argv[0], ": invalid option", option)
        sys.exit(1)


model_name = "{0}_{1}_{2}_{3}_{4}_{5}".format(
    "CNN", NUM_LAYERS, WINDOW_SIZE,
    LEARNING_RATE, DROPOUT_RATE, BATCH_SIZE
)

log_path = "./results/logs/" + model_name + ".txt"
model_path = "./results/models/" + model_name + ".ckpt"
log_file = open(log_path, "w")

print("hidden layers:", NUM_LAYERS)
print("hidden units:", NUM_HIDDEN[:NUM_LAYERS])
print("conv. filters:", CONV_FILTERS)
print("window size:", WINDOW_SIZE)
print("learning rate:", LEARNING_RATE)
print("dropout rate:", DROPOUT_RATE)
print("batch size:", BATCH_SIZE)
print()


# BUILDING GRAPH

echo("Creating placeholders...")

xs = tf.placeholder(tf.float32, [None, MAX_WORD_LEN, ALPHABET_SIZE])
ys = tf.placeholder(tf.float32, [None, NUM_GENDERS])
dropout = tf.placeholder(tf.float32)

echo("Creating model...")

model = CNNWordModel(
    xs, ys, dropout,
    CONV_FILTERS, WINDOW_SIZE, NUM_LAYERS, NUM_HIDDEN,
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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
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
                    dropout: DROPOUT_RATE
                }
            )

        val_loss, val_error = 0, 0
        val_batches = dataset.validation.words.shape[0] // 1024
        for i in range(val_batches):
            b_val_loss, b_val_error = sess.run(
                [model.loss, model.error],
                feed_dict={
                    xs: dataset.validation.words[1024 * i:1024 * (i + 1)],
                    ys: dataset.validation.genders[1024 * i:1024 * (i + 1)],
                    dropout: 0.0
                }
            )
            val_loss += b_val_loss / val_batches
            val_error += b_val_error / val_batches

        if val_error < best_val_error:
            best_epoch = epoch
            best_val_error = val_error
            saver.save(sess, model_path)

        log(log_file, "Epoch {:2d}:  error {:3.2f}%  loss {:.4f}".format(
            epoch, 100 * val_error, val_loss
        ))

    saver.restore(sess, model_path)

    test_loss, test_error = 0, 0
    test_batches = dataset.test.words.shape[0] // 1024
    for i in range(test_batches):
        b_test_loss, b_test_error = sess.run(
            [model.loss, model.error],
            feed_dict={
                xs: dataset.test.words[1024 * i:1024 * (i + 1)],
                ys: dataset.test.genders[1024 * i:1024 * (i + 1)],
                dropout: 0.0
            }
        )
        test_loss += b_test_loss / test_batches
        test_error += b_test_error / test_batches

    log(log_file)
    log(log_file, "Best epoch: {0}".format(best_epoch))
    log(log_file)
    log(log_file, "Test Set:  error {:3.2f}%  loss {:.4f}".format(
        100 * test_error, test_loss
    ))

log_file.close()
