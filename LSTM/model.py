import tensorflow as tf


class LSTMModel(tf.keras.Model):
    """Simple LSTM Model for grammar error detection

    Args:
        encoder (tf.keras.layers.Layer): a tf.keras.layers.Layer object serves as decoder
        bidirectional (boolean): a bool indicating whether build bidirectional model
    """

    def __init__(self,
                 encoder: tf.keras.layers.Layer,
                 bidirectional: bool = False):
        super().__init__()
        self._encoder = encoder
        self._pool1 = tf.keras.layers.AveragePooling1D()
        self._dropout1 = tf.keras.layers.Dropout(0.6)
        if bidirectional:
            self._decoder = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(50))
        else:
            self._decoder = tf.keras.layers.LSTM(50, return_sequences=True)
        self._pool2 = tf.keras.layers.AveragePooling1D()
        self._dropout2 = tf.keras.layers.Dropout(0.6)
        self._fc = tf.keras.layers.Dense(1, activation='sigmoid')


class SimpleModel(LSTMModel):

    def call(self, inputs):
        encoded = self._encoder(inputs)
        pooled1 = self._pool1(encoded)
        droped1 = self._dropout1(pooled1)
        decoded = self._decoder(droped1)
        pooled2 = self._pool2(decoded)
        droped2 = self._dropout2(pooled2)
        return self._fc(droped2)


class BertModel(LSTMModel):

    def call(self, inputs):
        encoded = self._encoder(**inputs)
        pooled1 = self._pool1(encoded[0])
        droped1 = self._dropout1(pooled1)
        decoded = self._decoder(droped1)
        pooled2 = self._pool2(decoded)
        droped2 = self._dropout2(pooled2)
        return self._fc(droped2)


class GPT2Model(LSTMModel):

    def call(self, inputs):
        encoded = self._encoder(**inputs)
        pooled1 = self._pool1(encoded[0])
        droped1 = self._dropout1(pooled1)
        decoded = self._decoder(droped1)
        pooled2 = self._pool2(decoded)
        droped2 = self._dropout2(pooled2)
        return self._fc(droped2)
