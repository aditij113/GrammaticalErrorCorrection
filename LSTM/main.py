import os

import tensorflow as tf
from absl import app, flags
from tensorflow.python.keras.backend import exp

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
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    tokenizer = utils.get_tokenizer(FLAGS.embedding)

    train_dataset, val_dataset, vocab_size = utils.create_train_val_dataset(
        FLAGS.data_path, FLAGS.label_path, tokenizer, FLAGS.batch_size,
        exp_name, FLAGS.val_split)

    device = '/gpu:0' if FLAGS.use_gpu else '/cpu:0'

    with tf.device(device):
        model = utils.get_model(FLAGS.embedding,
                                vocab_size,
                                FLAGS.output_dim,
                                bidirectional=FLAGS.bidirectional)
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr_rate)
        accuracy_tracker = tf.keras.metrics.BinaryAccuracy()
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_path, exp_name),
            verbose=1,
            save_best_only=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_path, 'tensorboard'), histogram_freq=1)

        model.compile(optimizer=optimizer,
                      loss=loss_function,
                      metrics=accuracy_tracker)
        model.fit(train_dataset,
                  epochs=FLAGS.epoch,
                  validation_data=val_dataset,
                  callbacks=[ckpt_callback, tensorboard_callback])


if __name__ == '__main__':
    app.run(main)
