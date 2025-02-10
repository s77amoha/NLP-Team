import csv
import os
import pickle

import matplotlib
import pandas as pd
import numpy as np
import tensorflow as tf

# noinspection PyUnresolvedReferences
from tensorflow.keras.models import load_model
# noinspection PyUnresolvedReferences
from tensorflow.keras.preprocessing.sequence import pad_sequences
# noinspection PyUnresolvedReferences
import tensorflow.keras as keras

from classical_nn.nn_train import f1_m, precision_m, recall_m

keras.config.enable_unsafe_deserialization()

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score



class Model:
    def __init__(self, name):
        self.name = name
        self.model_coarse = load_model(f"./models/coarse/{name}.keras", custom_objects={'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
        self.model_fine = load_model(f"./models/fine/{name}.keras", custom_objects={'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
        self.classes_coarse = open("subtask2_narratives.txt").read().split("\n")
        self.classes_fine = open("subtask2_subnarratives.txt").read().split("\n")

    def predict(self, texts, languages, raw=False):
        with open(f"./models/tokenizer_{self.name}.pickle", 'rb') as handle:
            tokenizer = pickle.load(handle)

        sequences = tokenizer.texts_to_sequences(texts)
        sequences_padded = pad_sequences(sequences, maxlen=924, padding='post')

        # Add language information
        language_val = np.array(languages)
        language_val[language_val == "EN"] = 1
        language_val[language_val == "PT"] = -1
        sequences_padded = np.column_stack((language_val, sequences_padded))
        sequences_padded = tf.convert_to_tensor(sequences_padded, dtype=float)

        predictions_coarse = self.model_coarse.predict(sequences_padded)
        predictions_fine = self.model_fine.predict(sequences_padded)

        predictions_coarse = [[1 if x >= 0.36 else 0 for x in element] for element in predictions_coarse]
        predictions_fine = [[1 if x >= 0.29 else 0 for x in element] for element in predictions_fine]

        if raw:
            return predictions_coarse, predictions_fine

        predictions_coarse = [[self.classes_coarse[i] for i in element] for element in predictions_coarse]
        predictions_fine = [[self.classes_fine[i] for i in element] for element in predictions_fine]

        return predictions_coarse, predictions_fine

    def score(self, texts, languages):
        pred_coarse, pred_fine = self.predict(texts, languages, raw=True)

        print("1. Coarse\n2. Fine\n")
        for (y_true, y_pred) in [(val_labels_coarse, pred_coarse),(val_labels_fine, pred_fine)]:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            # Calculate F1 score
            f1_micro = f1_score(y_true, y_pred, average='micro')
            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_weighted = f1_score(y_true, y_pred, average='weighted')

            # Calculate Accuracy
            accuracy = accuracy_score(y_true, y_pred)

            # Calculate Precision
            precision_micro = precision_score(y_true, y_pred, average='micro')
            precision_macro = precision_score(y_true, y_pred, average='macro')
            precision_weighted = precision_score(y_true, y_pred, average='weighted')

            # Calculate Recall
            recall_micro = recall_score(y_true, y_pred, average='micro')
            recall_macro = recall_score(y_true, y_pred, average='macro')
            recall_weighted = recall_score(y_true, y_pred, average='weighted')

            # Print the results
            print("-------")
            print(f"F1 Score (Micro): {f1_micro:.4f}")
            print(f"F1 Score (Macro): {f1_macro:.4f}")
            print(f"F1 Score (Weighted): {f1_weighted:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision (Micro): {precision_micro:.4f}")
            print(f"Precision (Macro): {precision_macro:.4f}")
            print(f"Precision (Weighted): {precision_weighted:.4f}")
            print(f"Recall (Micro): {recall_micro:.4f}")
            print(f"Recall (Macro): {recall_macro:.4f}")
            print(f"Recall (Weighted): {recall_weighted:.4f}")
            print("-------")


    def predict_and_write(self, texts, languages, ids, filename):
        """
        Predict labels for the test data and write the predictions to a file.
        """
        predictions_coarse, predictions_fine = self.predict(texts, languages)
        # write to file
        with open(filename, 'w', newline='') as file:
            tsv_writer = csv.writer(file, delimiter='\t')
            #tsv_writer.writerow(['ID', 'LANGUAGE', 'NARRATIVE', 'SUBNARRATIVE'])  # uncomment for a header
            for doc_id, coarse_pred, fine_pred in zip(ids, predictions_coarse, predictions_fine):
                tsv_writer.writerow([doc_id, doc_id[:2], ";".join(coarse_pred), ";".join(fine_pred)])


name = "new-3"
model = Model(name)

df_val = pd.read_csv("../target.csv")
texts = df_val["cleaned"].apply(lambda entry: entry.split()).to_list()
ids = df_val["filename"].to_list()
languages = df_val["language"].to_list()

output = f"./predictions/{name}_predictions.txt"
os.makedirs("./predictions", exist_ok=True)

#for scoring
classes_coarse = open("subtask2_narratives.txt").read().split("\n")
classes_fine = open("subtask2_subnarratives.txt").read().split("\n")
labels_coarse_val = df_val["narrative"].apply(lambda entry: entry.split(";")).to_list()
labels_fine_val = df_val["sub_narrative"].apply(lambda entry: entry.split(";")).to_list()
val_labels_coarse = tf.convert_to_tensor(
    [[1 if label in labels_coarse_val[i] else 0 for label in classes_coarse] for i in
     range(len(labels_coarse_val))])
val_labels_fine = tf.convert_to_tensor(
    [[1 if label in labels_fine_val[i] else 0 for label in classes_fine] for i in range(len(labels_fine_val))])

#model.score(texts, languages)
model.predict_and_write(texts, languages, ids, output)
