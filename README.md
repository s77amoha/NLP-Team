# Performance Comparison of Different ML Architectures for Narrative Classification

[Link to the GitHub Repository](https://github.com/s77amoha/NLP-Team)


# NLP-Team
NLP Team 23

## Introduction
This project analyzes the textual content of media articles to classify underlying narrative patterns. Using Natural Language Processing (NLP) techniques, we aim to detect and classify propaganda narratives in media articles related to climate change and the Ukraine conflict. The classification is based on a comprehensive narrative taxonomy, which includes multiple overlapping labels and finer-grained sub-narratives.

### Project Goals
- **Evaluate** different machine learning and deep learning models for narrative classification.
- **Compare** traditional machine learning models, deep learning models, and transformer-based models.
- **Analyze** model performance using F1-Score as the primary evaluation metric.
- **Identify** the most effective methods for detecting propaganda narratives in media.

## 2. Dataset
### 2.1 Statistical Measures
#### 2.1.1 Basic:
- **Corpus Size:**
  - Total number of documents.
  - Total number of tokens (words, punctuation, etc.).
- **Vocabulary Size:**
  - Number of unique words or terms in the corpus.
- **Average Document Length:**
  - Mean number of words or tokens per document.
#### 2.1.2 Lexical Diversity:
- **Type-Token Ratio (TTR)**
#### 2.1.3 Word Frequency Distribution:
- **Word Frequency Count**
#### 2.1.4 Syntactic and Grammatical Measures:
- **Part-of-Speech (POS) Distribution:**
  - Frequency and proportions of POS tags (e.g., nouns, verbs).

## Dataset
The dataset consists of labeled media articles in English and Portuguese, categorized under two primary domains:
- **Ukraine-Russia War (URW)**: Narratives like "Russia as the victim," "Discrediting Ukraine," and "Blaming the West."
- **Climate Change (CC)**: Narratives such as "Criticism of the climate movement," "Downplaying climate change," and "Hidden plots."

Data analysis includes:
- Narrative label distribution.
- Sentence length and vocabulary diversity.
- Topic prevalence via frequency distributions and word clouds.
- Exploratory Data Analysis (EDA) using PCA and t-SNE projections.
- **Lexical analysis** to determine term frequency across different narratives.
- **Semantic similarity analysis** to identify overlap between sub-narratives.

## 3. Methodology
### 3.1 Data Preprocessing
- Data cleansing:
  - Remove unnecessary characters.
  - Handle case sensitivity.
  - Remove stopwords.
  - Expand contractions (e.g., "don't" → "do not").
- **Tokenization**
- **Lemmatization**
- Handle **class imbalance** with weighted classes.
- **Data split and cross-validation**
- **Embedding normalization** for consistent representation across different models.

### 3.2 Feature Extraction
- **TF-IDF**
- **N-grams**
- **POS tagging**
- **Word Embeddings** or **Contextual Embeddings**
- **Named Entity Recognition (NER)** to identify key figures and institutions.
- **Dependency Parsing** to understand syntactic relationships between words.

### 3.3 Modeling Techniques
- **Traditional Approaches** (using BOW, TF-IDF):
  - Naive Bayes, Random Forests
- **Deep Learning Approaches** (using Word2Vec, GloVe):
  - Neural Networks (NN), Recurrent Neural Networks (RNN)
- **Transformer-based Approaches**:
  - BERT, Llama
- **Ensemble Learning** combining multiple models for improved robustness.

### 3.4 Evaluation
- Selecting appropriate metrics to compare the models.
- **Ablation studies** to evaluate the importance of each preprocessing step.
- **Error analysis** to identify common misclassifications.

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
- **XLNet and RoBERTa** as alternative transformer-based models for comparison.
  
## Expected Results
- Gain a better understanding of the corpus.
- Provide a clear framework to compare the performance of models and their pipelines.
- Assess which models generalize best across different narratives and sub-narratives.
- Establish whether multilingual data impacts classification performance.

## Evaluation Metrics
- **Basic Classification Metrics:**
  - Accuracy, Precision, Recall, F1-score, Confusion Matrix, etc.
- **Multi-Class Classification Metrics:**
  - Macro-averaged and Micro-averaged metrics.
- **Domain Adaptation Analysis** to determine model effectiveness across multiple propaganda themes.

### Training and Fine-Tuning
- **Feature extraction**: TF-IDF, Word2Vec embeddings.
- **Hyperparameter tuning**: Optimized for F1-score.
- **Oversampling and weighted loss functions** to handle class imbalances.
- **Evaluation metrics**: Accuracy, Precision, Recall, and F1-score.

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

  ## Challenges
- Identifying the best methods to clean the corpus.
- Selecting interesting models and pipelines to better understand the corpus.
- Choosing meaningful metrics to compare the results of different models.
- Handling **label imbalance** across multiple sub-narratives.
- Ensuring models do not **overfit** to high-frequency narratives.

## Conclusion
This study highlights the effectiveness of transformer-based models, especially BERT, in detecting and classifying propaganda narratives. While traditional and deep learning models show potential, transformers excel due to their ability to handle complex language structures and multi-label classification tasks.

## Future Work
- **Expand dataset** to include more languages and narratives.
- **Fine-tune Llama** for improved label consistency.
- **Develop real-time narrative detection pipelines** for news and social media monitoring.
- **Explore domain adaptation techniques** to generalize across different propaganda domains.
- **Investigate adversarial training** to improve robustness against manipulated content.
- **Improve model interpretability** by analyzing attention weights in transformer models.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
This project was conducted as part of the *Introduction to Natural Language Processing* course at the **University of Bonn, CAISA LAB (Winter Semester 2024/2025).
