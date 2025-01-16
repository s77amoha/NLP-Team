import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout
from keras.api.utils import to_categorical

# Load training data
train_df = pd.read_csv('training_data.csv')

# Load testing data
test_df = pd.read_csv('target.csv')

# Preprocess training data
train_texts = train_df['content'].tolist()
train_titles = train_df['title'].tolist()
train_labels_narrative = train_df['narrative'].apply(lambda x: x.split(';')).tolist()
train_labels_subnarrative = train_df['subnarrative'].apply(lambda x: x.split(';')).tolist()
train_languages = train_df['language'].tolist()

# Preprocess testing data
test_texts = test_df['content'].tolist()
test_titles = test_df['title'].tolist()
test_labels_narrative = test_df['narrative'].apply(lambda x: x.split(';')).tolist()
test_labels_subnarrative = test_df['subnarrative'].apply(lambda x: x.split(';')).tolist()
test_languages = test_df['language'].tolist()

# Vectorize texts using TfidfVectorizer
vectorizer = TfidfVectorizer(max_df=0.75, min_df=1, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_texts).toarray()
X_test = vectorizer.transform(test_texts).toarray()

# Binarize labels
mlb_narrative = MultiLabelBinarizer()
y_train_narrative = mlb_narrative.fit_transform(train_labels_narrative)
y_test_narrative = mlb_narrative.transform(test_labels_narrative)

mlb_subnarrative = MultiLabelBinarizer()
y_train_subnarrative = mlb_subnarrative.fit_transform(train_labels_subnarrative)
y_test_subnarrative = mlb_subnarrative.transform(test_labels_subnarrative)

# Encode languages
le = LabelEncoder()
y_train_lang = le.fit_transform(train_languages)
y_test_lang = le.transform(test_languages)
y_train_lang = to_categorical(y_train_lang)
y_test_lang = to_categorical(y_test_lang)

# Define the neural network model for language classification
model_lang = Sequential()
model_lang.add(Dense(512, input_shape=(X_train.shape[1],), activation='relu'))
model_lang.add(Dropout(0.5))
model_lang.add(Dense(len(le.classes_), activation='softmax'))
model_lang.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model for language classification
model_lang.fit(X_train, y_train_lang, epochs=10, batch_size=32, validation_split=0.2)

# Define the neural network model for narrative classification
model_narrative = Sequential()
model_narrative.add(Dense(512, input_shape=(X_train.shape[1],), activation='relu'))
model_narrative.add(Dropout(0.5))
model_narrative.add(Dense(y_train_narrative.shape[1], activation='sigmoid'))
model_narrative.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model for narrative classification
model_narrative.fit(X_train, y_train_narrative, epochs=30, batch_size=32, validation_split=0.2)

# Define the neural network model for subnarrative classification
model_subnarrative = Sequential()
model_subnarrative.add(Dense(512, input_shape=(X_train.shape[1],), activation='relu'))
model_subnarrative.add(Dropout(0.5))
model_subnarrative.add(Dense(y_train_subnarrative.shape[1], activation='sigmoid'))
model_subnarrative.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model for subnarrative classification
model_subnarrative.fit(X_train, y_train_subnarrative, epochs=40, batch_size=32, validation_split=0.2)

# Predict on testing data
pred_lang = model_lang.predict(X_test)
pred_narrative = model_narrative.predict(X_test)
pred_subnarrative = model_subnarrative.predict(X_test)

# Convert predictions to binary labels
pred_labels_lang = le.inverse_transform(pred_lang.argmax(axis=1))
pred_labels_narrative = mlb_narrative.inverse_transform(pred_narrative > 0.5)
pred_labels_subnarrative = mlb_subnarrative.inverse_transform(pred_subnarrative > 0.5)

def calculate_and_plot_accuracy(y_test_narrative, pred_narrative, y_test_subnarrative, pred_subnarrative, y_test_lang, pred_lang):
    # Convert predictions to binary labels for accuracy calculation
    pred_narrative_binary = (pred_narrative > 0.5).astype(int)
    pred_subnarrative_binary = (pred_subnarrative > 0.5).astype(int)
    pred_lang_binary = pred_lang.argmax(axis=1)

    # Calculate accuracy
    accuracy_narrative = accuracy_score(y_test_narrative, pred_narrative_binary)
    accuracy_subnarrative = accuracy_score(y_test_subnarrative, pred_subnarrative_binary)
    accuracy_lang = accuracy_score(y_test_lang.argmax(axis=1), pred_lang_binary)

    print(f"Accuracy for narrative labels: {accuracy_narrative}")
    print(f"Accuracy for subnarrative labels: {accuracy_subnarrative}")
    print(f"Accuracy for language: {accuracy_lang}")

    # Plot accuracy
    labels = ['Narrative', 'Subnarrative', 'Language']
    accuracies = [accuracy_narrative, accuracy_subnarrative, accuracy_lang]

    plt.bar(labels, accuracies, color=['blue', 'green', 'purple'])
    plt.xlabel('Label Type')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Neural Network Classifier')
    plt.ylim(0, 1)
    plt.show()

def calculate_and_print_precision(y_test_narrative, pred_narrative, y_test_subnarrative, pred_subnarrative, y_test_lang, pred_lang):
    # Convert predictions to binary labels for precision calculation
    pred_narrative_binary = (pred_narrative > 0.5).astype(int)
    pred_subnarrative_binary = (pred_subnarrative > 0.5).astype(int)
    pred_lang_binary = pred_lang.argmax(axis=1)

    # Calculate precision
    precision_narrative = precision_score(y_test_narrative, pred_narrative_binary, average='micro')
    precision_subnarrative = precision_score(y_test_subnarrative, pred_subnarrative_binary, average='micro')
    precision_lang = precision_score(y_test_lang.argmax(axis=1), pred_lang_binary, average='micro')

    print(f"Precision for narrative labels: {precision_narrative}")
    print(f"Precision for subnarrative labels: {precision_subnarrative}")
    print(f"Precision for language: {precision_lang}")

def calculate_and_print_recall(y_test_narrative, pred_narrative, y_test_subnarrative, pred_subnarrative, y_test_lang, pred_lang):
    # Convert predictions to binary labels for recall calculation
    pred_narrative_binary = (pred_narrative > 0.5).astype(int)
    pred_subnarrative_binary = (pred_subnarrative > 0.5).astype(int)
    pred_lang_binary = pred_lang.argmax(axis=1)

    # Calculate recall
    recall_narrative = recall_score(y_test_narrative, pred_narrative_binary, average='micro')
    recall_subnarrative = recall_score(y_test_subnarrative, pred_subnarrative_binary, average='micro')
    recall_lang = recall_score(y_test_lang.argmax(axis=1), pred_lang_binary, average='micro')

    print(f"Recall for narrative labels: {recall_narrative}")
    print(f"Recall for subnarrative labels: {recall_subnarrative}")
    print(f"Recall for language: {recall_lang}")

def calculate_and_print_f1(y_test_narrative, pred_narrative, y_test_subnarrative, pred_subnarrative, y_test_lang, pred_lang):
    # Convert predictions to binary labels for F1 score calculation
    pred_narrative_binary = (pred_narrative > 0.5).astype(int)
    pred_subnarrative_binary = (pred_subnarrative > 0.5).astype(int)
    pred_lang_binary = pred_lang.argmax(axis=1)

    # Calculate F1 score
    f1_narrative = f1_score(y_test_narrative, pred_narrative_binary, average='micro')
    f1_subnarrative = f1_score(y_test_subnarrative, pred_subnarrative_binary, average='micro')
    f1_lang = f1_score(y_test_lang.argmax(axis=1), pred_lang_binary, average='micro')

    print(f"F1 score for narrative labels: {f1_narrative}")
    print(f"F1 score for subnarrative labels: {f1_subnarrative}")
    print(f"F1 score for language: {f1_lang}")

def plot_misclassified_examples(test_titles, test_texts, test_labels_narrative, pred_labels_narrative, test_labels_subnarrative, pred_labels_subnarrative, test_languages, pred_labels_lang):
    misclassified_narrative = 0
    misclassified_subnarrative = 0
    misclassified_lang = 0

    for i in range(len(test_texts)):
        if set(pred_labels_narrative[i]) != set(test_labels_narrative[i]):
            misclassified_narrative += 1
        if set(pred_labels_subnarrative[i]) != set(test_labels_subnarrative[i]):
            misclassified_subnarrative += 1
        if pred_labels_lang[i] != test_languages[i]:
            misclassified_lang += 1

    labels = ['Narrative', 'Subnarrative', 'Language']
    misclassified_counts = [misclassified_narrative, misclassified_subnarrative, misclassified_lang]

    plt.bar(labels, misclassified_counts, color=['red', 'orange', 'purple'])
    plt.xlabel('Label Type')
    plt.ylabel('Number of Misclassified Examples')
    plt.title('Misclassified Examples of Neural Network Classifier')
    plt.show()

# Call the functions to calculate and plot metrics
calculate_and_plot_accuracy(y_test_narrative, pred_narrative, y_test_subnarrative, pred_subnarrative, y_test_lang, pred_lang)
calculate_and_print_precision(y_test_narrative, pred_narrative, y_test_subnarrative, pred_subnarrative, y_test_lang, pred_lang)
calculate_and_print_recall(y_test_narrative, pred_narrative, y_test_subnarrative, pred_subnarrative, y_test_lang, pred_lang)
calculate_and_print_f1(y_test_narrative, pred_narrative, y_test_subnarrative, pred_subnarrative, y_test_lang, pred_lang)
plot_misclassified_examples(test_titles, test_texts, test_labels_narrative, pred_labels_narrative, test_labels_subnarrative, pred_labels_subnarrative, test_languages, pred_labels_lang)

# Print misclassified examples
# for i in range(len(test_texts)):
#     if set(pred_labels_narrative[i]) != set(test_labels_narrative[i]) or set(pred_labels_subnarrative[i]) != set(test_labels_subnarrative[i]) or pred_labels_lang[i] != test_languages[i]:
#         print(f"Title: {test_titles[i]}")
#         print(f"Text: {test_texts[i]}")
#         print(f"True Narrative Labels: {test_labels_narrative[i]}")
#         print(f"Predicted Narrative Labels: {pred_labels_narrative[i]}")
#         print(f"True Subnarrative Labels: {test_labels_subnarrative[i]}")
#         print(f"Predicted Subnarrative Labels: {pred_labels_subnarrative[i]}")
#         print(f"True Language: {test_languages[i]}")
#         print(f"Predicted Language: {pred_labels_lang[i]}")
#         print("")
