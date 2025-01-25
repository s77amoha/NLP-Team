# NLP-Team
NLP Team 23

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

### 3.2 Feature Extraction
- **TF-IDF**
- **N-grams**
- **POS tagging**
- **Word Embeddings** or **Contextual Embeddings**

### 3.3 Modeling Techniques
- **Traditional Approaches** (using BOW, TF-IDF):
  - Naive Bayes, Random Forests
- **Deep Learning Approaches** (using Word2Vec, GloVe):
  - Neural Networks (NN), Recurrent Neural Networks (RNN)
- **Transformer-based Approaches**:
  - BERT

### 3.4 Evaluation
- Selecting appropriate metrics to compare the models.

## 4. Expected Results
- Gain a better understanding of the corpus.
- Provide a clear framework to compare the performance of models and their pipelines.

## 5. Evaluation Metrics
- **Basic Classification Metrics:**
  - Accuracy, Precision, Recall, F1-score, Confusion Matrix, etc.
- **Multi-Class Classification Metrics:**
  - Macro-averaged and Micro-averaged metrics.

## 6. Challenges
- Identifying the best methods to clean the corpus.
- Selecting interesting models and pipelines to better understand the corpus.
- Choosing meaningful metrics to compare the results of different models.
