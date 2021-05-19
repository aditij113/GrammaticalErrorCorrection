import tensorflow as tf


class SimpleLSTMModel(tf.keras.Model):
    """Simple LSTM Model for grammar error detection

    Args:
        input_dim (int): a int indicate the input size
        output_dim (int): a int indicate output size of embedding
        encoder (tf.keras.layers.Layer): a tf.keras.layers.Layer object serves as decoder
        bidirectional (boolean): a bool indicating whether build bidirectional model
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 encoder: tf.keras.layers.Layer = tf.keras.layers.Embedding,
                 bidirectional: bool = False):
        super().__init__()
        self._encoder = encoder(input_dim, output_dim)
        if bidirectional:
            self._decoder = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(50))
        else:
            self._decoder = tf.keras.layers.LSTM(50)
        self._fc = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        encoded = self._encoder(inputs)
        decoded = self._decoder(encoded)
        return self._fc(decoded)
