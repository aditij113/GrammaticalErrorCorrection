import json
import os
from typing import List, Sequence, Tuple

import tensorflow as tf
from nltk.tokenize import WordPunctTokenizer
from numpy.lib.npyio import save

if not os.path.exists('tokenizer'):
    os.mkdir('tokenizer')


def simple_tokenizer(raw_data: Sequence[str],
                     exp_name: str) -> Tuple[List[Sequence[int]], int]:
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
    padded_data = tf.keras.preprocessing.sequence.pad_sequences(pre_pad_data,
                                                                padding='post')

    save_path = os.path.join('tokenizer/', exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'simple_tokenizer.json'), 'w') as f:
        json.dump(tokenizer.to_json(), f)

    return padded_data, len(tokenizer.word_index) + 1
