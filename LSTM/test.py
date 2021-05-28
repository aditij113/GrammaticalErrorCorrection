import os

import tensorflow as tf
from absl import app, flags

import utils

flags.DEFINE_bool('bidirectional', False, 'Whether use bidirectional RNN')
flags.DEFINE_enum('embedding', 'simple', ['simple', 'bert', 'gpt2'],
                  'Which tokenizer to use')
flags.DEFINE_bool('use_gpu', True, 'Whethe use GPU for training.')
flags.DEFINE_integer('epoch', 50, 'Number of epochs to train')
flags.DEFINE_float('val_split', 0.2, 'Validation split for training')
flags.DEFINE_string('data_path', None, 'Path to the inputs')
flags.DEFINE_string('label_path', None, 'Path to labels')
flags.DEFINE_integer('batch_size', 32, 'Training batch size')
flags.DEFINE_integer('output_dim', 50, 'Output dimension of embedding layer')
flags.DEFINE_string('model_path', None, 'Path to checkpoint and saved model')
flags.DEFINE_float('lr_rate', 0.001, "Model's learning rate")

FLAGS = flags.FLAGS


def main(_):
    exp_name_temp = '_'.join(['{}' for _ in range(5)])
    exp_name = exp_name_temp.format(FLAGS.embedding, FLAGS.bidirectional,
                                    FLAGS.epoch, FLAGS.batch_size,
                                    FLAGS.lr_rate)

    model_path = os.path.join(FLAGS.model_path, exp_name)
    tokenizer = utils.get_test_tokenizer(FLAGS.embedding)

    data, labels, vocab_size = utils.create_test_dataset(
        FLAGS.data_path, FLAGS.label_path, tokenizer, exp_name)

    device = '/gpu:0' if FLAGS.use_gpu else '/cpu:0'

    with tf.device(device):
        model = utils.get_model(FLAGS.embedding,
                                vocab_size,
                                FLAGS.output_dim,
                                bidirectional=FLAGS.bidirectional)
        loss_function = tf.keras.losses.BinaryCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr_rate)

        model.compile(optimizer=optimizer,
                      loss=loss_function,
                      metrics=[
                          tf.keras.metrics.BinaryAccuracy(),
                          tf.keras.metrics.AUC(),
                          tf.keras.metrics.AUC(curve='pr')
                      ])
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_path, 'evaluate'))

        model.load_weights(os.path.join(model_path, 'saved_model'))

        model.evaluate(x=data, y=labels, callbacks=[tensorboard_callback])


if __name__ == '__main__':
    app.run(main)
