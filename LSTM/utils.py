from typing import Callable, Tuple

import numpy as np
import tensorflow as tf

import model
import token_util


def get_encoder(encoder):
    if encoder == 'default':
        return tf.keras.layers.Embedding


def get_model(model_name):
    if model_name == 'SimpleLSTM':
        return model.SimpleLSTMModel


def get_tokenizer(tokenizer_name):
    if tokenizer_name == 'simple':
        return token_util.simple_tokenizer


def create_train_val_dataset(
        dataset_path: str,
        label_path: str,
        tokenizer: Callable,
        batch_size: int,
        exp_name: str,
        val_split: float = 0.2) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """Create tf.data.Dataset for training with batch size

    Args:
        dataset_path (str): path to the sentences
        label_path (str): path to the labels
        tokenizer (Callable): tokenization function that takes in list of
            sentence and return tokenized input for model with padding and vocab.
            size
        batch_size (int): batch size for training
        exp_name (str): unique experiment name for saving
        val_split (float): portion of dataset to be splited as validation set

    Returns:
        train_dataset (tf.data.Dataset): dataset for training
        val_dataset (tf.data.Dataset): dataset for validation
        vocab_size (int): vocab size of dataset
    """

    with open(dataset_path, encoding='utf-8') as f:
        data_text = f.read()

    with open(label_path) as f:
        label_text = f.read()

    raw_data = data_text.splitlines()
    labels = np.array(label_text.splitlines()).astype(int)

    tokens, vocab_size = tokenizer(raw_data, exp_name)

    dataset = tf.data.Dataset.from_tensor_slices((tokens, labels))
    size = len(dataset)
    val_size = int(size * val_split)
    dataset = dataset.shuffle(size)
    val_dataset = dataset.take(val_size).batch(batch_size)
    train_dataset = dataset.skip(val_size).batch(batch_size)
    return train_dataset, val_dataset, vocab_size
