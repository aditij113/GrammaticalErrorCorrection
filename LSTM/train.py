import os
import pickle

import numpy as np
import tensorflow as tf
from absl import app, flags

import utils

flags.DEFINE_bool('bidirectional', False, 'Whether use bidirectional RNN')
flags.DEFINE_enum('embedding', 'simple', ['simple', 'bert', 'gpt2'],
                  'Which tokenizer to use')
flags.DEFINE_bool('use_gpu', True, 'Whethe use GPU for training.')
flags.DEFINE_integer('epoch', 50, 'Number of epochs to train')
flags.DEFINE_float('val_split', 0.2, 'Validation split for training')
flags.DEFINE_string('train_data', None, 'Path to the train inputs')
flags.DEFINE_string('train_label', None, 'Path to the train labels')
flags.DEFINE_string('test_data', None, 'Path to the test inputs')
flags.DEFINE_string('test_label', None, 'Path to the test labels')
flags.DEFINE_integer('batch_size', 32, 'Training batch size')
flags.DEFINE_integer('output_dim', 50, 'Output dimension of embedding layer')
flags.DEFINE_string('model_path', None, 'Path to checkpoint and saved model')
flags.DEFINE_float('lr_rate', 0.001, "Model's learning rate")
flags.DEFINE_integer('input_length', 500,
                     'How long the input sequence should be')

FLAGS = flags.FLAGS


def main(_):
    exp_name_temp = '_'.join(['{}' for _ in range(5)])
    exp_name = exp_name_temp.format(FLAGS.embedding, FLAGS.bidirectional,
                                    FLAGS.epoch, FLAGS.batch_size,
                                    FLAGS.lr_rate)

    model_path = os.path.join(FLAGS.model_path, exp_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    train_tokenizer = utils.get_tokenizer(FLAGS.embedding)
    train_dataset, val_dataset, vocab_size = utils.create_train_val_dataset(
        FLAGS.train_data, FLAGS.train_label, train_tokenizer, FLAGS.batch_size,
        exp_name, FLAGS.val_split, FLAGS.input_length)

    test_tokenizer = utils.get_test_tokenizer(FLAGS.embedding)
    test_dataset, test_y, _ = utils.create_test_dataset(
        FLAGS.test_data, FLAGS.test_label, test_tokenizer, exp_name,
        FLAGS.batch_size, FLAGS.input_length)

    device = '/gpu:0' if FLAGS.use_gpu else '/cpu:0'

    with tf.device(device):
        model = utils.get_model(FLAGS.embedding,
                                vocab_size,
                                FLAGS.output_dim,
                                bidirectional=FLAGS.bidirectional)
        loss_function = tf.keras.losses.BinaryCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr_rate)
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_path, 'saved_model'),
            verbose=1,
            save_best_only=True,
            save_weights_only=True)
        train_tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_path, 'train'), histogram_freq=1)
        model.compile(optimizer=optimizer,
                      loss=loss_function,
                      metrics=[
                          tf.keras.metrics.BinaryAccuracy(),
                          tf.keras.metrics.AUC(),
                          tf.keras.metrics.AUC(curve='pr')
                      ])
        model.fit(train_dataset,
                  epochs=FLAGS.epoch,
                  validation_data=val_dataset,
                  callbacks=[ckpt_callback, train_tensorboard_callback])
        print(model.summary())

        predict = np.array([])
        for dset in test_dataset:
            predict = np.concatenate((predict, model.predict(dset).ravel()))
        print(predict)
        print(test_y)
        with open(os.path.join(model_path, 'eval.pickle'), 'wb') as f:
            pickle.dump({'predict': predict, 'label': test_y}, f)


if __name__ == '__main__':
    app.run(main)
