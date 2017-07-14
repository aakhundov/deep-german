import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from read_data import nouns_to_one_hot
from rnn_word_model import RNNWordModel


MAX_WORD_LEN = 31
ALPHABET_SIZE = 31
NUM_GENDERS = 3


# parsing hyperparameters from model name

model_name = "BasicLSTMCell_2_0.01_0.5_128"

if len(sys.argv) > 1:
    model_name = sys.argv[1]

model_tokens = model_name.split("_")
model_path = "./results/models/" + model_name + ".ckpt"

CELL_TYPE = getattr(rnn, model_tokens[0])
NUM_LAYERS = int(model_tokens[1])
LEARNING_RATE = float(model_tokens[2])
DROPOUT_RATE = float(model_tokens[3])
NUM_HIDDEN = [128, 128, 128]


# creating placeholders and model

xs = tf.placeholder(tf.float32, [None, MAX_WORD_LEN, ALPHABET_SIZE])
ys = tf.placeholder(tf.float32, [None, NUM_GENDERS])
seq = tf.placeholder(tf.int32, [None])
dropout = tf.placeholder(tf.float32)

model = RNNWordModel(
    xs, ys, seq, dropout,
    CELL_TYPE, NUM_LAYERS, NUM_HIDDEN,
    tf.train.AdamOptimizer(LEARNING_RATE)
)


# running a session

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, model_path)

    while True:
        noun = input("your word: ").lower()

        if noun == "":
            break
        else:
            try:
                one_hot, seq_len = nouns_to_one_hot([noun])
                prediction = sess.run(
                    model.prediction,
                    feed_dict={
                        xs: one_hot,
                        seq: seq_len,
                        dropout: 0.0
                    }
                )

                gender_id = np.asscalar(np.argmax(prediction[0]))
                article = ["der", "die", "das"][gender_id]

                print("prediction:", article, noun.capitalize())
                print("probs:", "[{:.2f}, {:.2f}, {:.2f}]".format(*prediction[0]))

            except KeyError:
                print("non-german characters")

            finally:
                print()
