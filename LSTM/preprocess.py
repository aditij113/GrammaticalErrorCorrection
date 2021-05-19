import os
import random

from absl import app, flags

flags.DEFINE_string('corr_sentence', None, 'correct sentences file')
flags.DEFINE_string('incorr_sentence', None, 'incorrect sentence file')
flags.DEFINE_string('output_path', None, 'data output path')
flags.DEFINE_float('pos_rate', 0.5, 'probability of correct sentence is taken')

FLAGS = flags.FLAGS


def main(_):
    prob = FLAGS.pos_rate

    with open(FLAGS.corr_sentence, 'r') as f:
        corr = f.readlines()

    with open(FLAGS.incorr_sentence, 'r') as f:
        incorr = f.readlines()

    data, label = [], []
    for corr_sent, incorr_sent in zip(corr, incorr):
        if random.random() < prob:
            data.append(corr_sent)
            label.append("1\n")
        else:
            data.append(incorr_sent)
            label.append("0\n")

    with open(os.path.join(FLAGS.output_path, 'data.txt'), 'w') as f:
        f.writelines(data)

    with open(os.path.join(FLAGS.output_path, 'label.txt'), 'w') as f:
        f.writelines(label)


if __name__ == '__main__':
    app.run(main)
