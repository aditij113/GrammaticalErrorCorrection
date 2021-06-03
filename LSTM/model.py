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
        if bidirectional:
            self._decoder = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(120))
        else:
            self._decoder = tf.keras.layers.LSTM(120, return_sequences=True)

        self._pool = tf.keras.layers.AveragePooling1D(padding='same')
        self._dropout = tf.keras.layers.Dropout(0.5)
        self._flatten = tf.keras.layers.Flatten()
        self._fc = tf.keras.layers.Dense(1, activation='sigmoid')


class SimpleModel(LSTMModel):

    def call(self, inputs):
        encoded = self._encoder(inputs)
        decoded = self._decoder(encoded)
        pooled = self._pool(decoded)
        droped = self._dropout(pooled)
        flated = self._flatten(droped)
        return self._fc(flated)


class BertModel(LSTMModel):

    def call(self, inputs):
        encoded = self._encoder(**inputs)
        decoded = self._decoder(encoded[0])
        pooled = self._pool(decoded)
        droped = self._dropout(pooled)
        flated = self._flatten(droped)
        return self._fc(flated)


class GPT2Model(LSTMModel):

    def call(self, inputs):
        encoded = self._encoder(**inputs)
        decoded = self._decoder(encoded[0])
        pooled = self._pool(decoded)
        droped = self._dropout(pooled)
        flated = self._flatten(droped)
        return self._fc(flated)
