import os
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
from gensim.models import Word2Vec

# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Input, Embedding, Dense, GlobalMaxPooling1D, Dropout, concatenate, Lambda
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Model as KerasModel, Sequential
# noinspection PyUnresolvedReferences
from tensorflow.keras import backend as K
# noinspection PyUnresolvedReferences
from tensorflow.keras.preprocessing.text import Tokenizer
# noinspection PyUnresolvedReferences
from tensorflow.keras.preprocessing.sequence import pad_sequences


def recall_m(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def create_model_coarse(num_labels, max_doc_length, vocab_size, embedding_matrix, embedding_dim):
    input_layer = Input(shape=(1 + max_doc_length,), name="Input")
    input_lang = Lambda(lambda x: x[:, :1], name="lang_input")(input_layer)
    input_text = Lambda(lambda x: x[:, 1:], name="text_input")(input_layer)
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_doc_length,
                                weights=[embedding_matrix], name="Word2Vec", trainable=False)(input_text)

    sequential_part1 = Sequential([
        Dense(500, activation='relu', name="Dense_1"),
        Dense(500, activation='relu', name="Dense_2"),
        Dropout(0.3, name="Dropout_1"),
        Dense(350, activation='relu', name="Dense_3"),
        Dense(350, activation='relu', name="Dense_4"),
        GlobalMaxPooling1D(name="GlobalMaxPooling"),
    ], name="NN1")(embedding_layer)

    language_info = Dense(1, name="lang")(input_lang)
    concat = concatenate([language_info, sequential_part1], name="Concat")
    sequential_part2 = Sequential([
        Dense(200, activation='relu', name="Dense_5"),
        Dropout(0.2, name="Dropout_2"),
        Dense(200, activation='relu', name="Dense_6"),
        Dense(100, activation='relu', name="Dense_7"),
        Dense(64, activation='elu', name="Dense_8"),
        Dense(num_labels, activation="sigmoid", name="Output")
    ], name="NN2")(concat)

    model = KerasModel(input_layer, sequential_part2, name="TextClassifier_Coarse")

    #Default start lr for adam is 0.001
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
    model.summary()

    return model


def create_model_fine(num_labels, max_doc_length, vocab_size, embedding_matrix, embedding_dim):
    input_layer = Input(shape=(1 + max_doc_length,), name="Input")
    input_lang = Lambda(lambda x: x[:, :1], name="lang_input")(input_layer)
    input_text = Lambda(lambda x: x[:, 1:], name="text_input")(input_layer)
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_doc_length,
                                weights=[embedding_matrix], name="Word2Vec", trainable=False)(input_text)

    sequential_part1 = Sequential([
        Dense(500, activation='relu', name="Dense_1"),
        Dense(500, activation='relu', name="Dense_2"),
        Dropout(0.3, name="Dropout_1"),
        Dense(350, activation='relu', name="Dense_3"),
        Dense(350, activation='relu', name="Dense_4"),
        GlobalMaxPooling1D(name="GlobalMaxPooling"),
    ], name="NN1")(embedding_layer)

    language_info = Dense(1, name="lang")(input_lang)
    concat = concatenate([language_info, sequential_part1], name="Concat")
    sequential_part2 = Sequential([
        Dense(200, activation='relu', name="Dense_5"),
        Dropout(0.2, name="Dropout_2"),
        Dense(200, activation='relu', name="Dense_6"),
        Dense(100, activation='relu', name="Dense_7"),
        Dense(64, activation='elu', name="Dense_8"),
        Dense(num_labels, activation="sigmoid", name="Output")
    ], name="NN2")(concat)

    model = KerasModel(input_layer, sequential_part2, name="TextClassifier_Fine")

    # Default start lr for adam is 0.001
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
    model.summary()

    return model


def train(texts, labels_coarse, labels_fine, language_train,
          val_texts, val_labels_coarse, val_labels_fine, language_val,
          all_classes_coarse, all_classes_fine, name,
          max_doc_length=924, vocab_size=21543, embedding_dim=100):

    # Tokenizer
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts + texts_val)
    sequences = tokenizer.texts_to_sequences(texts)
    sequence_pad = pad_sequences(sequences, maxlen=max_doc_length, padding='post')
    val_sequences = tokenizer.texts_to_sequences(val_texts)
    val_sequence_pad = pad_sequences(val_sequences, maxlen=max_doc_length, padding='post')

    # Convert to training data
    labels_coarse = tf.convert_to_tensor([[1 if label in labels_coarse[i] else 0 for label in all_classes_coarse] for i in range(len(labels_coarse))])
    labels_fine = tf.convert_to_tensor([[1 if label in labels_fine[i] else 0 for label in all_classes_fine] for i in range(len(labels_fine))])
    val_labels_coarse = tf.convert_to_tensor([[1 if label in val_labels_coarse[i] else 0 for label in all_classes_coarse] for i in range(len(val_labels_coarse))])
    val_labels_fine = tf.convert_to_tensor([[1 if label in val_labels_fine[i] else 0 for label in all_classes_fine] for i in range(len(val_labels_fine))])

    # Add language information
    language_train = np.array(language_train)
    language_train[language_train == "en"] = 1
    language_train[language_train == "pt"] = -1
    language_val = np.array(language_val)
    language_val[language_val == "en"] = 1
    language_val[language_val == "pt"] = -1

    sequence_pad = np.column_stack((language_train, sequence_pad))
    val_sequence_pad = np.column_stack((language_val, val_sequence_pad))
    sequence_pad = tf.convert_to_tensor(sequence_pad, dtype=float)
    val_sequence_pad = tf.convert_to_tensor(val_sequence_pad, dtype=float)

    with open(f'./models/tokenizer_{name}.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #Word2Vec Embedding
    word2vec_model = Word2Vec(texts, vector_size=embedding_dim, window=10, min_count=1, workers=4)

    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]

    # Compile models
    print("Compiling sub-models...")
    coarse_submodel = create_model_coarse(len(all_classes_coarse), max_doc_length, vocab_size, embedding_matrix, embedding_dim)
    fine_submodel = create_model_fine(len(all_classes_fine), max_doc_length, vocab_size, embedding_matrix, embedding_dim)

    # Train
    print("Training models...")

    path_coarse = "./models/coarse/"
    path_fine = "./models/fine/"
    os.makedirs(path_coarse, exist_ok=True)
    os.makedirs(path_fine, exist_ok=True)

    # b. checkpoint
    filepath_checkpoint_coarse = path_coarse + f"{name}.keras"
    coarse_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath_checkpoint_coarse,
                                                           save_weights_only=False,
                                                           save_best_only=True)

    filepath_checkpoint_fine = path_fine + f"{name}.keras"
    fine_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath_checkpoint_fine,
                                                         save_weights_only=False,
                                                         save_best_only=True)

    print("Coarse Model....")
    coarse_submodel.fit(sequence_pad,
                        labels_coarse,
                        epochs=30,
                        verbose=1,
                        callbacks=[coarse_checkpoint],
                        validation_data=(val_sequence_pad, val_labels_coarse),
                        )

    print("Fine Model....")
    fine_submodel.fit(sequence_pad,
                      labels_fine,
                      epochs=40,
                      verbose=1,
                      callbacks=[fine_checkpoint],
                      validation_data=(val_sequence_pad, val_labels_fine),
                      )


if __name__ == '__main__':
    classes_coarse = open("subtask2_narratives.txt").read().split("\n")
    classes_fine = open("subtask2_subnarratives.txt").read().split("\n")

    df = pd.concat([pd.read_csv("../target.csv"), pd.read_csv("../training_data.csv")], ignore_index=True).drop_duplicates(subset="title", keep="first")

    texts = df["content"].apply(lambda entry: entry.split()).to_numpy()
    languages = df["language"].to_numpy()
    labels_coarse = df["narrative"].apply(lambda entry: entry.split(";")).to_numpy()
    labels_fine = df["subnarrative"].apply(lambda entry: entry.split(";")).to_numpy()

    print(len(texts))

    train_indices = np.load("./train_indices.npy", allow_pickle=True)
    val_indices = np.load("./val_indices.npy", allow_pickle=True)

    texts_train = list([texts[i] for i in train_indices])
    labels_coarse_train = list([labels_coarse[i] for i in train_indices])
    labels_fine_train = list([labels_fine[i] for i in train_indices])
    lang_train = list([languages[i] for i in train_indices])
    texts_val = list([texts[i] for i in val_indices])
    labels_coarse_val = list([labels_coarse[i] for i in val_indices])
    labels_fine_val = list([labels_fine[i] for i in val_indices])
    lang_val = list([languages[i] for i in val_indices])

    train(texts_train,
          labels_coarse_train,
          labels_fine_train,
          lang_train,
          texts_val,
          labels_coarse_val,
          labels_fine_val,
          lang_val,
          classes_coarse,
          classes_fine,
          "attempt3-window10")
