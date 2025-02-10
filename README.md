# Performance Comparison of Different ML Architectures for Narrative Classification

## Introduction
This project analyzes the textual content of media articles to classify underlying narrative patterns. Using Natural Language Processing (NLP) techniques, we aim to detect and classify propaganda narratives in media articles related to climate change and the Ukraine conflict. The classification is based on a comprehensive narrative taxonomy, which includes multiple overlapping labels and finer-grained sub-narratives.

### Project Goals
- **Evaluate** different machine learning and deep learning models for narrative classification.
- **Compare** traditional machine learning models, deep learning models, and transformer-based models.
- **Analyze** model performance using F1-Score as the primary evaluation metric.
- **Identify** the most effective methods for detecting propaganda narratives in media.

## Dataset
The dataset consists of labeled media articles in English and Portuguese, categorized under two primary domains:
- **Ukraine-Russia War (URW)**: Narratives like "Russia as the victim," "Discrediting Ukraine," and "Blaming the West."
- **Climate Change (CC)**: Narratives such as "Criticism of the climate movement," "Downplaying climate change," and "Hidden plots."

Data analysis includes:
- Narrative label distribution.
- Sentence length and vocabulary diversity.
- Topic prevalence via frequency distributions and word clouds.
- Exploratory Data Analysis (EDA) using PCA and t-SNE projections.

## Models Used
We evaluate five different models categorized into three main groups:

### 1. **Traditional Machine Learning Models**
- **Naive Bayes** (Multinomial Naive Bayes with Laplace smoothing)
- **Random Forest** (Ensemble learning with multiple decision trees)

### 2. **Deep Learning Models**
- **Neural Networks (NN)** (Multi-Layer Perceptron with Word2Vec embeddings)
- **LSTM (Long Short-Term Memory Networks)** (Captures sequential dependencies in text)

### 3. **Transformer-Based Models**
- **BERT** (Pre-trained transformer model fine-tuned for classification)
- **Llama** (Large-scale transformer model supporting long-context inputs)

## Methodology
### Data Preprocessing
- Lowercasing and expanding contractions.
- Removal of numbers, special characters, and stop words.
- Tokenization and lemmatization.
- Handling class imbalances through oversampling.
- Vectorization using TF-IDF and word embeddings.

### Training and Fine-Tuning
- **Feature extraction**: TF-IDF, Word2Vec embeddings.
- **Hyperparameter tuning**: Optimized for F1-score.
- **Oversampling and weighted loss functions** to handle class imbalances.
- **Evaluation metrics**: Accuracy, Precision, Recall, and F1-score.

## Experimental Setup
### Evaluation Metrics
- **Accuracy**: Measures overall correctness.
- **Precision**: Measures positive predictive value.
- **Recall**: Measures completeness of predictions.
- **F1-Score**: Harmonic mean of Precision and Recall.
- **Confusion Matrix**: Analyzes classification errors.

### Model Performance Comparison
| Model            | Narrative F1-Score | Sub-Narrative F1-Score |
|-----------------|------------------|---------------------|
| Naive Bayes     | 0.50             | 0.34                |
| Random Forest   | 0.34             | 0.11                |
| Classical NN    | 0.50             | 0.30                |
| LSTM            | 0.42             | 0.27                |
| BERT            | **0.55**         | **0.38**            |
| Llama           | 0.52             | 0.31                |

## Results and Discussion
- **BERT outperformed other models**, achieving the highest F1-score in both narrative and sub-narrative classification.
- **Llama showed strong results** but required more post-processing to prevent label drift.
- **Traditional ML models (Naive Bayes, Random Forest) struggled** due to their limited ability to capture nuanced propaganda narratives.
- **Deep learning models (NN, LSTM) improved performance** but were less effective than transformer-based models.
- **Handling class imbalance was crucial** to improve recall for rare sub-narratives.

## Conclusion
This study highlights the effectiveness of transformer-based models, especially BERT, in detecting and classifying propaganda narratives. While traditional and deep learning models show potential, transformers excel due to their ability to handle complex language structures and multi-label classification tasks.

## Future Work
- **Expand dataset** to include more languages and narratives.
- **Fine-tune Llama** for improved label consistency.
- **Develop real-time narrative detection pipelines** for news and social media monitoring.
- **Explore domain adaptation techniques** to generalize across different propaganda domains.

## Contributors
- Name Surname (student id number)


## Acknowledgments
This project was conducted as part of the *Introduction to Natural Language Processing* course at the **University of Bonn, CAISA LAB (Winter Semester 2024/2025)**.

