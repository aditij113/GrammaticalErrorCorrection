import json
import os
from typing import List, Sequence, Tuple

import tensorflow as tf
import transformers
from nltk.tokenize import WordPunctTokenizer

if not os.path.exists('tokenizer'):
    os.mkdir('tokenizer')


def simple_tokenizer(raw_data: Sequence[str], exp_name: str,
                     max_length: int) -> Tuple[List[Sequence[int]], int]:
    """Use simple tokenization to convert texts to int ids with padding.

    Args:
        raw_data (Sequence[str]): sequence of string of raw data
        exp_name (str): unique experiment name for saving

    Returns:
        padded_data (List): a list of sequence int ids with 0 padding at the
            end.
        vocab_size (int): a int indicating the number of vocab by the tokenizer
    """

    punc_tokenizer = WordPunctTokenizer()
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<UNK>')
    token_data = [punc_tokenizer.tokenize(sent) for sent in raw_data]
    tokenizer.fit_on_texts(token_data)

    pre_pad_data = tokenizer.texts_to_sequences(token_data)
    padded_data = tf.keras.preprocessing.sequence.pad_sequences(
        pre_pad_data, maxlen=max_length, padding='post')

    save_path = os.path.join('tokenizer/', exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'simple_tokenizer.json'), 'w') as f:
        json.dump(tokenizer.to_json(), f)

    return padded_data, len(tokenizer.word_index) + 1


def simple_test_tokenizer(raw_data: Sequence[str], exp_name: str,
                          max_length: int) -> Tuple[List[Sequence], int]:
    save_path = os.path.join('tokenizer/', exp_name)
    with open(os.path.join(save_path, 'simple_tokenizer.json')) as f:
        json_text = json.load(f)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_text)
    punc_tokenizer = WordPunctTokenizer()
    token_data = [punc_tokenizer.tokenize(sent) for sent in raw_data]
    pre_pad_data = tokenizer.texts_to_sequences(token_data)
    padded_data = tf.keras.preprocessing.sequence.pad_sequences(
        pre_pad_data, maxlen=max_length, padding='post')
    return padded_data, len(tokenizer.word_index) + 1


def bert_tokenizer(raw_data: Sequence[str], exp_name: str,
                   max_length: int) -> Tuple[List[Sequence[int]], int]:
    punc_tokenizer = WordPunctTokenizer()
    token_data = [['[CLS]'] + punc_tokenizer.tokenize(sent) + ['[SEP]']
                  for sent in raw_data]
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    pad_data = tokenizer(token_data,
                         padding='max_length',
                         truncation=True,
                         max_length=max_length,
                         is_split_into_words=True)
    return dict(pad_data), tokenizer.vocab_size


def gpt2_tokenizer(raw_data: Sequence[str], exp_name: str,
                   max_length: int) -> Tuple[List[Sequence[int]], int]:
    punc_tokenizer = WordPunctTokenizer()
    token_data = [punc_tokenizer.tokenize(sent) for sent in raw_data]
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.unk_token
    pad_data = tokenizer(token_data,
                         padding='max_length',
                         max_length=max_length,
                         truncation=True,
                         is_split_into_words=True)
    return dict(pad_data), tokenizer.vocab_size
