# Necessary and residual imports
import nltk
from nltk.corpus import udhr
import pandas as pd
import regex
import nltk.tokenize.casual
import tensorflow as tf
import gc

import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('books_labels.tar.gz', compression='gzip', header=0, sep=',', quotechar='"', on_bad_lines='skip')
spanish_df = df[df['language'] == 'SPA']
portuguese_df = df[df['language'] == 'POR']

spanish_df = spanish_df.sort_values(by=['file_name_translation', 'id'])[df['file_name_translation'] == 'Nueva Biblia de las Américas (NBLA)'] # https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.reddit.com/r/Reformed/comments/xoye82/most_accurate_spanish_bible_translations_spanish/&ved=2ahUKEwj7qbSKtbuFAxV4IEQIHbm6B_UQFnoECCwQAQ&usg=AOvVaw2YZJ8eHalVoloIoaaATROH
spanish_df = spanish_df.set_index('id')
spanish_df = spanish_df.drop(columns=['books_labels.csv', 'codebook', 'language', 'book_file_name', 'file_name_translation', 'source', 'year', 'genre', 'genre-multilabel', 'testament', 'division'])
portuguese_df = portuguese_df.sort_values(by=['file_name_translation', 'id'])[df['file_name_translation'] == 'SF_2009-01-20_POR_ACF_(PORTUGUESE CORRIGIDA FIEL (1753_1995))']
portuguese_df = portuguese_df.set_index('id')
portuguese_df = portuguese_df.drop(columns=['books_labels.csv', 'codebook', 'language', 'book_file_name', 'file_name_translation', 'source', 'year', 'genre', 'genre-multilabel', 'testament', 'division'])
spanish_df = spanish_df.rename(columns={'text':'spanish_text'})
portuguese_df = portuguese_df.rename(columns={'text':'portuguese_text'})

multi_df = spanish_df.join(portuguese_df)
multi_df = multi_df.dropna()

print("Spanish:", multi_df['spanish_text']['b.1CH.001.011'],"Portuguese:", multi_df['portuguese_text']['b.1CH.001.011'])

spanish_list = multi_df['spanish_text'].to_list()
portuguese_list = multi_df['portuguese_text'].to_list()

print('Spanish:', spanish_list[11], 'Portuguese:', portuguese_list[11])
spanish_list = [regex.sub(r'\([a-zA-z0-9]\)', '', item) for item in spanish_list]
spanish_corpus = [nltk.tokenize.casual_tokenize(item) for item in spanish_list]
print(spanish_corpus[0])
portuguese_corpus = [nltk.tokenize.casual_tokenize(item) for item in portuguese_list]
print(portuguese_corpus[0])
del df
del spanish_list, portuguese_list, spanish_df, portuguese_df, multi_df
gc.collect()
spanish_encoder_texts, spanish_input_texts, spanish_target_texts = [], [], []
portuguese_encoder_texts, portuguese_input_texts, portuguese_target_texts = [], [], []
spanish_vocabulary = set()
portuguese_vocabulary = set()
spanish_start_token = '[SPSTART]'
portuguese_start_token = '[POSTART]'
stop_token = '[END]'
unknown_token = '[UNK]'
pad_token = '[PAD]'
spanish_vocabulary.add(spanish_start_token)
spanish_vocabulary.add(stop_token)
spanish_vocabulary.add(unknown_token)
spanish_vocabulary.add(pad_token)
portuguese_vocabulary.add(portuguese_start_token)
portuguese_vocabulary.add(stop_token)
portuguese_vocabulary.add(unknown_token)
portuguese_vocabulary.add(pad_token)

for spanish_text in spanish_corpus:
    spanish_encoder_texts.append([spanish_start_token] + spanish_text + [stop_token])
    spanish_input_texts.append([spanish_start_token] + spanish_text)
    spanish_target_texts.append(spanish_text + [stop_token])
    for char in spanish_text:
        if char not in spanish_vocabulary:
            spanish_vocabulary.add(char)

for portuguese_text in portuguese_corpus:
    portuguese_encoder_texts.append([portuguese_start_token] + portuguese_text + [stop_token])
    portuguese_input_texts.append([portuguese_start_token] + portuguese_text)
    portuguese_target_texts.append(portuguese_text + [stop_token])
    for char in portuguese_text:
        if char not in portuguese_vocabulary:
            portuguese_vocabulary.add(char)

unified_vocabulary = spanish_vocabulary.union(portuguese_vocabulary)

print(len(spanish_vocabulary), len(unified_vocabulary), len(portuguese_vocabulary))
print(spanish_encoder_texts[0], portuguese_input_texts[0], portuguese_target_texts[0])
spanish_vocabulary = sorted(spanish_vocabulary)
portuguese_vocabulary = sorted(portuguese_vocabulary)

# Define maxima
spanish_vocab_size = len(spanish_vocabulary)
portuguese_vocab_size = len(portuguese_vocabulary)
unified_vocab_size = len(unified_vocabulary)
max_spanish_seq_length = max([len(txt) for txt in spanish_target_texts])
max_portuguese_seq_length = max([len(txt) for txt in portuguese_target_texts])
max_unified_seq_length = max(max_spanish_seq_length, max_portuguese_seq_length)

# Create indicies
spanish_token_index = dict([(token, i) for i, token in
                          enumerate(spanish_vocabulary)])
portuguese_token_index = dict([(token, i) for i, token in
                          enumerate(portuguese_vocabulary)])
unified_token_index = dict([(token, i) for i, token in
                          enumerate(unified_vocabulary)])
reverse_spanish_token_index = dict([(i, token) for token, i in
                          spanish_token_index.items()])
reverse_portuguese_token_index = dict([(i, token) for token, i in
                          portuguese_token_index.items()])
reverse_unified_token_index = dict([(i, token) for token, i in
                          unified_token_index.items()])
print(max_spanish_seq_length, max_unified_seq_length, max_portuguese_seq_length)
def convert_text_to_indices(texts, token_index, max_seq_length, pad_token='[PAD]'):
    data = np.zeros((len(texts), max_seq_length),
                    dtype='int32')
    for i, text in enumerate(texts):
        for t, token in enumerate(text):
            data[i, t] = token_index[token]
        for t in range(len(text), max_seq_length):
            data[i, t] = token_index[pad_token]
    
    return data
import numpy as np

# Convert sentences to numpy arrays
spanish_encoder_input_data = convert_text_to_indices(spanish_encoder_texts, unified_token_index, max_spanish_seq_length+1)
spanish_decoder_input_data = convert_text_to_indices(spanish_input_texts, unified_token_index, max_spanish_seq_length)
spanish_decoder_target_data = convert_text_to_indices(spanish_target_texts, unified_token_index, max_spanish_seq_length)

portuguese_encoder_input_data = convert_text_to_indices(portuguese_encoder_texts, unified_token_index, max_portuguese_seq_length+1)
portuguese_decoder_input_data = convert_text_to_indices(portuguese_input_texts, unified_token_index, max_portuguese_seq_length)
portuguese_decoder_target_data = convert_text_to_indices(portuguese_target_texts, unified_token_index, max_portuguese_seq_length)
batch_size = 64
epochs = 20
num_neurons = 256
num_layers = 4
d_model = 256
dff = 512
num_heads = 8
dropout_rate = 0.1
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
print(f'Running on a TPU w/{tpu.num_accelerators()["TPU"]} cores')
from transformers import *
transformer = Transformer(num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=unified_vocab_size,
    target_vocab_size=unified_vocab_size,
    dropout_rate=dropout_rate)
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])
dataset = tf.data.Dataset.from_tensor_slices(((spanish_encoder_input_data, portuguese_decoder_input_data), portuguese_decoder_target_data))
batched_dataset = dataset.batch(batch_size)
transformer.fit(batched_dataset, epochs=40)