import csv
import pickle
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import load_model
# noinspection PyUnresolvedReferences
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Model:
    def __init__(self, name):
        self.name = name
        self.model_coarse = load_model(f"./models/coarse/{name}.keras")
        self.model_fine = load_model(f"./models/fine/{name}.keras")
        self.classes_coarse = open("subtask2_narratives.txt").read().split("\n")
        self.classes_fine = open("subtask2_subnarratives.txt").read().split("\n")

    def predict(self, texts):
        with open(f"./models/tokenizer_{self.name}.pickle", 'rb') as handle:
            tokenizer = pickle.load(handle)

        sequences = tokenizer.texts_to_sequences(texts)
        sequences_padded = pad_sequences(sequences, maxlen=924, padding='post')

        predictions_coarse = self.model_coarse.predict(sequences_padded)
        predictions_fine = self.model_fine.predict(sequences_padded)

        predictions_coarse = [self.classes_coarse[i] for i in predictions_coarse]
        predictions_fine = [self.classes_fine[i] for i in predictions_fine]

        return predictions_coarse, predictions_fine

    def predict_and_write(self, texts, ids, filename):
        """
        Predict labels for the test data and write the predictions to a file.
        """
        predictions_coarse, predictions_fine = self.predict(texts)
        # write to file
        with open(filename, 'w', newline='') as file:
            tsv_writer = csv.writer(file, delimiter='\t')
            # tsv_writer.writerow(['ID', 'NARRATIVE', 'SUBNARRATIVE'])  # uncomment for a header
            for doc_id, coarse_pred, fine_pred in zip(ids, predictions_coarse, predictions_fine):
                tsv_writer.writerow([doc_id, ";".join(coarse_pred), ";".join(fine_pred)])


#load model
model = Model("attempt2")

#TODO
