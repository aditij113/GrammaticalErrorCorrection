import os

import tensorflow as tf
from absl import app, flags

import utils

flags.DEFINE_enum('model', 'SimpleLSTM', ['SimpleLSTM'], 'Model to be used')
flags.DEFINE_bool('bidirectional', False, 'Whether create bidirectional model')
flags.DEFINE_enum(
    'encoder', 'default', ['default'],
    'Which encoder to use in the model. Default one use tf.keras.layers.Embedding'
)
flags.DEFINE_enum('tokenizer', 'simple', ['simple'], 'Which tokenizer to use')
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
    exp_name = exp_name_temp.format(FLAGS.model, FLAGS.tokenizer, FLAGS.epoch,
                                    FLAGS.batch_size, FLAGS.lr_rate)

    model_path = os.path.join(FLAGS.model_path, exp_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    tokenizer = utils.get_tokenizer(FLAGS.tokenizer)

    train_dataset, val_dataset, vocab_size = utils.create_train_val_dataset(
        FLAGS.data_path, FLAGS.label_path, tokenizer, FLAGS.batch_size,
        FLAGS.val_split)

    device = '/gpu:0' if FLAGS.use_gpu else '/cpu:0'

    print(train_dataset)
    with tf.device(device):
        encoder = utils.get_encoder(FLAGS.encoder)
        model = utils.get_model(FLAGS.model)(vocab_size,
                                             FLAGS.output_dim,
                                             encoder=encoder,
                                             bidirectional=FLAGS.bidirectional)
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr_rate)
        accuracy_tracker = tf.keras.metrics.BinaryAccuracy()
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_path, exp_name),
            verbose=1,
            save_best_only=True)

        model.compile(optimizer=optimizer,
                      loss=loss_function,
                      metrics=accuracy_tracker)
        model.fit(train_dataset,
                  epochs=FLAGS.epoch,
                  validation_data=val_dataset,
                  callbacks=ckpt_callback)
        model.save(os.path.join(model_path, 'SavedModel'))


if __name__ == '__main__':
    app.run(main)
