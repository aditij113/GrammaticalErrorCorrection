import os
import random

from absl import app, flags

flags.DEFINE_enum('type', 'train', ['train', 'test'],
                  'which dataset to process')
flags.DEFINE_string('corr_sentence', None, 'correct sentences file')
flags.DEFINE_string('incorr_sentence', None, 'incorrect sentence file')
flags.DEFINE_string('output_path', None, 'data output path')
flags.DEFINE_string('input_path', None, 'input path to test dataset')
flags.DEFINE_float('pos_rate', 0.5, 'probability of correct sentence is taken')

FLAGS = flags.FLAGS


def train_data_process():
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

    with open(os.path.join(FLAGS.output_path, 'train_data.txt'), 'w') as f:
        f.writelines(data)

    with open(os.path.join(FLAGS.output_path, 'train_label.txt'), 'w') as f:
        f.writelines(label)


def test_data_process():
    with open(FLAGS.input_path, 'r') as f:
        data_with_label = f.readlines()

    data, label = [], []
    for line in data_with_label:
        label.append(line[-2:])
        data.append(line[:-3] + '\n')

    with open(os.path.join(FLAGS.output_path, 'test_data.txt'), 'w') as f:
        f.writelines(data)

    with open(os.path.join(FLAGS.output_path, 'test_label.txt'), 'w') as f:
        f.writelines(label)


def main(_):
    if FLAGS.type == 'train':
        flags.mark_flags_as_required(['corr_sentence', 'incorr_sentence'])
        train_data_process()
    else:
        flags.mark_flag_as_required('input_path')
        test_data_process()


if __name__ == '__main__':
    app.run(main)
