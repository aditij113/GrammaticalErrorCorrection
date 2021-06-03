from typing import Callable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
import transformers

import model
import token_util


def get_encoder(encoder, input_dim, output_dim):
    if encoder == 'simple':
        return tf.keras.layers.Embedding(input_dim, output_dim)
    if encoder == 'bert':
        encode = transformers.TFBertMainLayer(
            transformers.BertConfig.from_pretrained('bert-base-uncased'))
        encode.trainable = False
        return encode

    if encoder == 'gpt2':
        encode = transformers.TFGPT2MainLayer(
            transformers.GPT2Config.from_pretrained('gpt2'))
        encode.trainable = False
        return encode


def get_model(embedding, input_dim, output_dim, bidirectional):
    embedding_layer = get_encoder(embedding, input_dim, output_dim)
    if embedding == 'simple':
        return model.SimpleModel(embedding_layer, bidirectional)
    if embedding == 'bert':
        return model.BertModel(embedding_layer, bidirectional)
    if embedding == 'gpt2':
        embedding_layer.config.pad_token_id = embedding_layer.config.eos_token_id
        return model.GPT2Model(embedding_layer, bidirectional)


def get_tokenizer(tokenizer_name):
    if tokenizer_name == 'simple':
        return token_util.simple_tokenizer
    if tokenizer_name == 'bert':
        return token_util.bert_tokenizer
    if tokenizer_name == 'gpt2':
        return token_util.gpt2_tokenizer


def get_test_tokenizer(tokenizer_name):
    if tokenizer_name == 'simple':
        return token_util.simple_test_tokenizer
    if tokenizer_name == 'bert':
        return token_util.bert_tokenizer
    if tokenizer_name == 'gpt2':
        return token_util.gpt2_tokenizer


def create_train_val_dataset(
        dataset_path: str,
        label_path: str,
        tokenizer: Callable,
        batch_size: int,
        exp_name: str,
        val_split: float = 0.2,
        seed: int = 51,
        max_length: int = 500) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
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
        seed (int): integer seed for random shuffle. To force each split is consistent.

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
    labels = np.array(label_text.splitlines()).astype(int).reshape((-1, 1))

    tokens, vocab_size = tokenizer(raw_data, exp_name, max_length)

    dataset = tf.data.Dataset.from_tensor_slices((tokens, labels))
    size = len(dataset)
    val_size = int(size * val_split)
    dataset = dataset.shuffle(size, seed=seed)
    val_dataset = dataset.take(val_size).batch(batch_size)
    train_dataset = dataset.skip(val_size).batch(batch_size)
    return train_dataset, val_dataset, vocab_size


def create_test_dataset(dataset_path: str, label_path: str, tokenizer: Callable,
                        exp_name: str, batch_size: int, max_length: int):
    with open(dataset_path, encoding='utf-8') as f:
        data_text = f.read()

    with open(label_path) as f:
        label_text = f.read()

    raw_data = data_text.splitlines()
    labels = np.array(label_text.splitlines()).astype(int)

    tokens, vocab_size = tokenizer(raw_data, exp_name, max_length)
    dataset = tf.data.Dataset.from_tensor_slices(tokens).batch(batch_size)

    return dataset, labels, vocab_size
