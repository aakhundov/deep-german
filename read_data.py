import collections
import numpy as np


clean_nouns_path = "./data/clean_nouns.txt"
german_chars = "abcdefghijklmnopqrstuvwxyzßäöü"
allowed_gender_labels = ["m", "f", "n", "pl"]

Datasets = collections.namedtuple(
    'Datasets', ['train', 'validation', 'test']
)


# this is adapted from the original TensorFlow DataSet class in
# tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet
class DataSet(object):
    def __init__(self, words, genders, seq_length):
        self._num_examples = words.shape[0]

        self._words = words
        self._genders = genders
        self._seq_length = seq_length
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def words(self):
        return self._words

    @property
    def genders(self):
        return self._genders

    @property
    def seq_length(self):
        return self._seq_length

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._words = self.words[perm0]
            self._genders = self.genders[perm0]
            self._seq_length = self.seq_length[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            words_rest_part = self._words[start:self._num_examples]
            genders_rest_part = self._genders[start:self._num_examples]
            seq_length_rest_part = self._seq_length[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._words = self.words[perm]
                self._genders = self.genders[perm]
                self._seq_length = self.seq_length[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            words_new_part = self._words[start:end]
            genders_new_part = self._genders[start:end]
            seq_length_new_part = self._seq_length[start:end]
            return np.concatenate((words_rest_part, words_new_part), axis=0), \
                np.concatenate((genders_rest_part, genders_new_part), axis=0), \
                np.concatenate((seq_length_rest_part, seq_length_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._words[start:end], self._genders[start:end], self._seq_length[start:end]


def read_data_sets(soft_labels=False, validation_ratio=0.1):
    max_length, gender_labels = 30, 4
    codes = {c: i for i, c in enumerate(german_chars)}

    with open(clean_nouns_path, "r") as f:
        lines = [line[:-1] for line in f.readlines()]

    rand_state = np.random.get_state()
    np.random.seed(12345)
    np.random.shuffle(lines)
    np.random.set_state(rand_state)

    one_hot_words = np.zeros([len(lines), max_length, len(codes)], dtype=np.float32)
    one_hot_genders = np.zeros([len(lines), gender_labels], dtype=np.float32)
    seq_length = np.zeros([len(lines)], dtype=np.int64)

    for i, line in enumerate(lines):
        noun, genders = line.split("\t")
        gender_counts = [int(g) for g in genders.split(",")]

        for j, c in enumerate(noun):
            one_hot_words[i, j, codes[c]] = 1.0

        if soft_labels:
            total_counts = sum(gender_counts)
            for j, g in enumerate(gender_counts):
                one_hot_genders[i, j] = g / total_counts
        else:
            one_hot_genders[i, np.argmax(gender_counts)] = 1.0

        seq_length[i] = len(noun)

    test_size = int(len(lines) * 0.1)
    validation_size = int(len(lines) * validation_ratio)
    train_size = len(lines) - validation_size - test_size

    train = DataSet(one_hot_words[:train_size], one_hot_genders[:train_size], seq_length[:train_size])
    test = DataSet(one_hot_words[-test_size:], one_hot_genders[-test_size:], seq_length[-test_size:])
    validation = DataSet(
        one_hot_words[train_size:train_size+validation_size],
        one_hot_genders[train_size:train_size+validation_size],
        seq_length[train_size:train_size+validation_size]
    )

    return Datasets(train=train, validation=validation, test=test)


def reconstruct_batch(one_hot_words, one_hot_genders, seq_length):
    words, genders = [], []
    for i in range(len(one_hot_words)):
        words.append("".join([
            german_chars[np.asscalar(np.argmax(one_hot_words[i, j]))]
            for j in range(seq_length[i])
        ]))
        genders.append({
            g: one_hot_genders[i, j]
            for j, g in enumerate(allowed_gender_labels)
        })

    return words, genders


if __name__ == "__main__":

    sets = read_data_sets()

    print(
        "training set:",
        "words", sets.train.words.shape,
        ", genders", sets.train.genders.shape,
        ", seq. length", sets.train.seq_length.shape
    )
    print(
        "validation set",
        "words", sets.validation.words.shape,
        ", genders", sets.validation.genders.shape,
        ", seq. length", sets.validation.seq_length.shape
    )
    print(
        "test set",
        "words", sets.test.words.shape,
        ", genders", sets.test.genders.shape,
        ", seq. length", sets.test.seq_length.shape
    )
    print()

    one_hot_xs, one_hot_ys, seq_len = sets.test.next_batch(20)
    real_words, gender_maps = reconstruct_batch(one_hot_xs, one_hot_ys, seq_len)

    print("Reconstructed random mini-batch of 20 words:")
    print("--------------------------------------------")
    for rw, gm in zip(real_words, gender_maps):
        print(rw, gm)
