import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

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

# Vectorize texts
vectorizer = CountVectorizer(max_df=0.75, min_df=1)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

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

# Train Naive Bayes classifier for language
nb_lang = MultinomialNB(alpha=1.5, fit_prior=True)
nb_lang.fit(X_train, y_train_lang)

# Train Naive Bayes classifier for narrative labels
nb_narrative = OneVsRestClassifier(MultinomialNB(alpha=1.5, fit_prior=True))
nb_narrative.fit(X_train, y_train_narrative)

# Train Naive Bayes classifier for subnarrative labels
nb_subnarrative = OneVsRestClassifier(MultinomialNB(alpha=1.5, fit_prior=True))
nb_subnarrative.fit(X_train, y_train_subnarrative)

# Predict on testing data
pred_lang = nb_lang.predict(X_test)
pred_narrative = nb_narrative.predict(X_test)
pred_subnarrative = nb_subnarrative.predict(X_test)

# Convert predictions back to label format
pred_labels_lang = le.inverse_transform(pred_lang)
pred_labels_narrative = mlb_narrative.inverse_transform(pred_narrative)
pred_labels_subnarrative = mlb_subnarrative.inverse_transform(pred_subnarrative)

def calculate_and_plot_accuracy(y_test_narrative, pred_narrative, y_test_subnarrative, pred_subnarrative, y_test_lang, pred_lang):
    # Calculate accuracy
    accuracy_narrative = accuracy_score(y_test_narrative, pred_narrative)
    accuracy_subnarrative = accuracy_score(y_test_subnarrative, pred_subnarrative)
    accuracy_lang = accuracy_score(y_test_lang, pred_lang)

    print(f"Accuracy for narrative labels: {accuracy_narrative}")
    print(f"Accuracy for subnarrative labels: {accuracy_subnarrative}")
    print(f"Accuracy for language: {accuracy_lang}")

    # Plot accuracy
    labels = ['Narrative', 'Subnarrative', 'Language']
    accuracies = [accuracy_narrative, accuracy_subnarrative, accuracy_lang]

    plt.bar(labels, accuracies, color=['blue', 'green', 'purple'])
    plt.xlabel('Label Type')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Naive Bayes Classifier')
    plt.ylim(0, 1)
    plt.show()

def calculate_and_print_precision(y_test_narrative, pred_narrative, y_test_subnarrative, pred_subnarrative, y_test_lang, pred_lang):
    # Calculate precision
    precision_narrative = precision_score(y_test_narrative, pred_narrative, average='micro')
    precision_subnarrative = precision_score(y_test_subnarrative, pred_subnarrative, average='micro')
    precision_lang = precision_score(y_test_lang, pred_lang, average='micro')

    print(f"Precision for narrative labels: {precision_narrative}")
    print(f"Precision for subnarrative labels: {precision_subnarrative}")
    print(f"Precision for language: {precision_lang}")

def calculate_and_print_recall(y_test_narrative, pred_narrative, y_test_subnarrative, pred_subnarrative, y_test_lang, pred_lang):
    # Calculate recall
    recall_narrative = recall_score(y_test_narrative, pred_narrative, average='micro')
    recall_subnarrative = recall_score(y_test_subnarrative, pred_subnarrative, average='micro')
    recall_lang = recall_score(y_test_lang, pred_lang, average='micro')

    print(f"Recall for narrative labels: {recall_narrative}")
    print(f"Recall for subnarrative labels: {recall_subnarrative}")
    print(f"Recall for language: {recall_lang}")

def calculate_and_print_f1(y_test_narrative, pred_narrative, y_test_subnarrative, pred_subnarrative, y_test_lang, pred_lang):
    # Calculate F1 score
    f1_narrative = f1_score(y_test_narrative, pred_narrative, average='micro')
    f1_subnarrative = f1_score(y_test_subnarrative, pred_subnarrative, average='micro')
    f1_lang = f1_score(y_test_lang, pred_lang, average='micro')

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
    plt.title('Misclassified Examples of Naive Bayes Classifier')
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

