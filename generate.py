import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from read_data import nouns_to_one_hot
from rnn_word_model import RNNWordModel


MAX_WORD_LEN = 31
ALPHABET_SIZE = 31
NUM_GENDERS = 3

MIN_GEN_WORD_LEN = 3
MAX_GEN_WORD_LEN = 20
GEN_WORDS_PER_GENDER = 10000

german_chars = "abcdefghijklmnopqrstuvwxyzßäöü"
total_chars = len(german_chars)

noun_endings = {
    "masculine": ["ant", "anz", "ast", "er", "ich", "eich",
                  "ig", "eig", "ling", "or", "us", "ismus"],
    "feminine": ["anz", "e", "ei", "enz", "heit", "ie", "in", "ik", "keit",
                 "nis", "schaft", "sion", "sis", "tion", "tät", "ung", "ur"],
    "neutral": ["a", "chen", "lein", "en", "il", "in", "ing",
                "it", "ma", "ment", "nis", "tum", "um", ]
}


def generate_nouns(noun_ending, count=1, min_len=1, max_len=60):
    end_len = len(noun_ending)
    min_len = max(min_len, end_len+1)
    max_len = max(max_len, min_len+1)

    result = []
    for _ in range(count):
        length = np.random.randint(low=min_len-end_len, high=max_len-end_len)
        word = "".join([german_chars[np.random.randint(total_chars)] for _ in range(length)])
        word += noun_ending
        result.append(word)

    return result


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

np.random.seed(12345)

with tf.Session() as sess:
    saver.restore(sess, model_path)

    for gender in noun_endings.keys():
        print(gender)
        print("-------------------------------------")
        for ending in noun_endings[gender]:
            nouns = generate_nouns(
                ending, GEN_WORDS_PER_GENDER,
                MIN_GEN_WORD_LEN, MAX_GEN_WORD_LEN
            )

            one_hot, seq_len = nouns_to_one_hot(nouns)
            prediction = sess.run(
                model.prediction,
                feed_dict={
                    xs: one_hot,
                    seq: seq_len,
                    dropout: 0.0
                }
            )

            counts = [0, 0, 0]
            for g in np.argmax(prediction, axis=1):
                counts[g] += 1

            fractions = [c / sum(counts) * 100 for c in counts]
            print("-{:<10}{:<10.2f}{:<10.2f}{:<10.2f}".format(
                ending, *fractions
            ))
        print()
