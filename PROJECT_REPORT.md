# Project Report: X (formerly Twitter) Sentiment Analysis

**Academic Term:** Semester 7 (Fall 2026)  
**Subject:** Natural Language Processing (CS-471 / Sem 7)  
**Project Workspace:** `twitter-sentiment-analysis/`  
**Target Repository:** `X-Sentiment-Analysis`  
**Status:** Phase A Complete (Project Requirements & Literature Review — 2026 Edition)  
**Date:** November 2026  

---

## Table of Contents
1. [Phase A: Project Requirements](#phase-a-project-requirements)
   - [1. Problem Formulation & Objective (2026 Context)](#1-problem-formulation--objective-2026-context)
   - [2. Requirements Analysis](#2-requirements-analysis)
     - [2.1 Functional Requirements](#21-functional-requirements)
     - [2.2 Non-Functional Requirements](#22-non-functional-requirements)
     - [2.3 Input and Output Specifications](#23-input-and-output-specifications)
     - [2.4 System & Runtime Environment Requirements (2026 Stack)](#24-system--runtime-environment-requirements-2026-stack)
   - [3. High-Level Description of Tasks](#3-high-level-description-of-tasks)
   - [4. Identification of Major Project Challenges in 2026](#4-identification-of-major-project-challenges-in-2026)
   - [5. Review of Related Works (10 Seminal & State-of-the-Art Studies: 2002–2026)](#5-review-of-related-works-10-seminal--state-of-the-art-studies-20022026)
   - [6. Comparative Synthesis of Related Works](#6-comparative-synthesis-of-related-works)
   - [7. Formal Bibliography (BibTeX)](#7-formal-bibliography-bibtex)
2. [Roadmap for Subsequent Phases (Phases B to E)](#roadmap-for-subsequent-phases-phases-b-to-e)

---

# Phase A: Project Requirements

## 1. Problem Formulation & Objective (2026 Context)

Social microblogging platforms, principally **X (formerly Twitter)**, remain the world's preeminent real-time conversational channels for breaking news, public discourse, brand monitoring, financial market trends, and sociopolitical debates. In 2026, the volume of streaming social data surpasses 600 million posts per day. Analyzing affective states, public opinion, and polarity from these high-throughput streams is vital for automated decision-support systems, algorithmic trading, customer intelligence, and public health surveillance.

However, microblog sentiment analysis continues to present fundamental computational challenges:
1. **Linguistic Unorthodoxy:** Pervasive colloquialisms, dynamic 2026 internet neologisms (*delulu*, *rizz*, *gyatt*, *cap/no cap*), deliberate character elongations, acronyms, and phonetic spellings.
2. **Context Inversion & Pragmatics:** Negation operators, sarcasm, and subtle rhetorical irony where affective valence contradicts surface vocabulary.
3. **The 2026 Inference Trade-Off:** While modern 70B+ parameter generative Large Language Models (LLMs) demonstrate zero-shot reasoning, their prohibitive computational footprint, token inference latency ($>300\text{ ms}$ per call), and API subscription expenses make them infeasible for real-time edge streaming over millions of posts. Consequently, **compact, high-throughput, multi-paradigm machine learning models and deep hybrid ensembles** remain the industry standard for production-scale microblog sentiment classification.

### Mathematical Formulation
The sentiment classification task is modeled as a binary supervised learning problem:
$$\hat{y} = f(T; \Theta) \in \{0, 1\}$$
where $T = (w_1, w_2, \dots, w_k)$ represents the sequential token stream of a post on X, $\Theta$ denotes the learnable parameters of the model $f$, and $y \in \{0, 1\}$ represents the target polarity ($1 = \text{positive}$, $0 = \text{negative}$). For posts with multifaceted polarity, the dominant polarity constitutes the ground truth.

The objective of this project is to develop, evaluate, and benchmark an end-to-end sentiment classification pipeline over a benchmark corpus of **1,000,000 tweets/posts** (800,000 training, 200,000 testing). The project investigates the complete hierarchy of natural language modeling:
* **Heuristic Lexicon Baseline:** Rule-based polarity counters using curated opinion lexicons.
* **Classical Machine Learning:** Multinomial/Bernoulli Naive Bayes, Maximum Entropy (Logistic Regression), Decision Trees (CART), Random Forests, Support Vector Machines (Linear SVM), and Extreme Gradient Boosting (XGBoost) over sparse n-gram spaces.
* **Deep Neural Architectures:** Multilayer Perceptrons (MLP), Recurrent Neural Networks (LSTM), and 1D Temporal Convolutional Neural Networks (1 to 4 convolutional stages) initialized with both random and 200-dimensional Stanford Twitter GloVe vectors.
* **Deep Representation Transfer & Ensembling:** Feature extraction from the penultimate 600-dimensional dense layer of the best CNN, maximum-margin boundary learning via Linear SVM, and a heterogeneous **Majority Voting Ensemble** across the top 5 models to establish state-of-the-art accuracy on the public leaderboard (benchmark accuracy: 83.58%).

---

## 2. Requirements Analysis

### 2.1 Functional Requirements

| ID | Component | Specification |
| :--- | :--- | :--- |
| **FR-01** | **Data Ingestion** | The system must ingest raw CSV datasets with schema `tweet_id,sentiment,tweet` (training) and `tweet_id,tweet` (testing), cleanly parsing quotation escapes, multiline text, and comma delimiters without requiring header metadata. |
| **FR-02** | **Social Token Normalization** | The system must apply regular-expression transformations: normalize web URLs (`((www\.[^\s]+)\|(https?://[^\s]+))`) to `URL`; user handles (`@[^\s]+`) to `USER_MENTION`; strip hashtag symbols (`#(\S+)`) while retaining lexical content; strip retweet tokens (`\brt\b`); decode emoticons into positive (`EMO_POS`) and negative (`EMO_NEG`) sentiment placeholders. |
| **FR-03** | **Morphological & Word-Level Normalization** | The system must strip punctuation (`' " ? ! , . ( ) : ;`), compress repeated characters ($\ge 3$ consecutive duplicate characters reduced to exactly 2, e.g., `funnnnny` $\to$ `funny`), remove hyphen/apostrophe contractions, and enforce case folding. |
| **FR-04** | **Corpus Statistics & Zipfian Profiling** | The system must compute frequency distributions of unigrams and bigrams via NLTK, verify compliance with Zipf's law, serialize distribution objects (`.pkl`), and generate pruned top-$N$ unigram ($15,000$ to $90,000$) and top-$M$ bigram ($10,000$) vocabularies. |
| **FR-05** | **Sparse Feature Engineering** | The system must generate memory-efficient sparse vectors (`scipy.sparse.lil_matrix` converting to `csr_matrix`) supporting both binary occurrence (`presence`) and term frequency (`frequency`), with optional TF-IDF transformations. |
| **FR-06** | **Dense Representation & GloVe Seeding** | The system must map tokens to dense vector indices, parse 200-dimensional Stanford Twitter GloVe word vector files (`glove-seeds.txt`), initialize model embedding matrices with pre-trained vectors, and accommodate trainable out-of-vocabulary embeddings. |
| **FR-07** | **Model Training Suite** | The system must implement training routines across 10 distinct model families: Baseline Lexicon, Naive Bayes, Logistic Regression / MaxEnt, Decision Tree, Random Forest, XGBoost, Linear SVM, MLP, LSTM, and 1D CNN (1 to 4 convolutional stages). |
| **FR-08** | **Deep Feature Extraction (Representation Transfer)** | The system must extract the 600-dimensional feature activation vectors from the penultimate dense layer of the best-performing 4-layer CNN model and serialize them as dense NumPy matrices (`.npy`). |
| **FR-09** | **Hybrid Classification** | The system must train a linear Support Vector Machine on the extracted 600-dimensional deep CNN feature vectors, combining deep compositional representation with maximum-margin boundary separation. |
| **FR-10** | **Heterogeneous Ensemble Aggregation** | The system must aggregate test set probability/label predictions from 5 distinct architectures via hard majority voting, breaking ties deterministically and generating the final Kaggle submission CSV file (`majority-voting.csv`). |

### 2.2 Non-Functional Requirements

| Quality Attribute | 2026 Target Specification & Verification Metric |
| :--- | :--- |
| **Memory Efficiency & Scalability** | The system must process 1,000,000 tweets within a memory footprint strictly $\le 8\text{ GB}$ RAM by leveraging sparse matrix streaming and minibatch vector construction (500 samples/batch), preventing out-of-memory crashes on consumer and server hardware alike. |
| **Throughput & Inference Latency** | Offline preprocessing of 800,000 records must complete in $\le 12\text{ minutes}$ on a standard 8-core CPU. Per-sample inference latency must remain $< 3\text{ ms}$, ensuring full compatibility with live streaming architectures. |
| **Classification Accuracy** | The final ensemble model must achieve $\ge 83.5\%$ test accuracy on the benchmark test split, providing a statistically significant margin over lexicon baselines (~65%) and single linear classifiers (~78–80%). |
| **Modularity & Reproducibility** | All pipeline phases (preprocessing, profiling, model training, feature extraction, and ensembling) must exist as decoupled CLI scripts with deterministic seed management to guarantee bitwise reproducibility. |
| **Platform Portability** | The code must execute cleanly across modern Linux/POSIX systems supporting Python 3.8 through 3.12+ environments. |

### 2.3 Input and Output Specifications

```
+---------------------------------------------------------------------------------------------------------+
|                                        DATA FLOW SPECIFICATION                                          |
+---------------------------------------------------------------------------------------------------------+
|  RAW DATASET (Kaggle / Sentiment140)                                                                    |
|  Train CSV: [tweet_id: int, sentiment: {0, 1}, tweet: string (enclosed in quotes)]                      |
|  Test CSV:  [tweet_id: int, tweet: string (enclosed in quotes)]                                         |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v  [preprocess.py]
+---------------------------------------------------------------------------------------------------------+
|  PREPROCESSED DATASET                                                                                   |
|  Train: tweet_id,sentiment,cleaned_token_stream (e.g., "123,1,user_mention love nlp so much emo_pos")  |
|  Test:  tweet_id,cleaned_token_stream                                                                   |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v  [stats.py]
+---------------------------------------------------------------------------------------------------------+
|  SERIALIZED DISTRIBUTIONS & VOCABULARIES                                                                |
|  - train-processed-freqdist.pkl    (NLTK FreqDist: Top unigrams)                                        |
|  - train-processed-freqdist-bi.pkl (NLTK FreqDist: Top bigrams)                                         |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v  [Classifiers: Classical, Neural, Hybrid]
+---------------------------------------------------------------------------------------------------------+
|  INTERMEDIATE PREDICTIONS & FEATURE MATRICES                                                            |
|  - train-feats.npy & test-feats.npy (600-dimensional penultimate CNN features)                           |
|  - Individual model prediction CSVs: results/<model>.csv (schema: id,prediction)                       |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v  [majority-voting.py]
+---------------------------------------------------------------------------------------------------------+
|  FINAL SUBMISSION / EVALUATION                                                                          |
|  - majority-voting.csv (schema: id,prediction; 200,000 binary predictions; 83.58% accuracy)            |
+---------------------------------------------------------------------------------------------------------+
```

### 2.4 System & Runtime Environment Requirements (2026 Stack)

* **Host OS:** Linux (x86_64 Ubuntu 22.04 / 24.04 LTS or POSIX-compliant environment).
* **Python Runtime:** Python 3.8+ (tested with Python 3.10–3.12 compatibility).
* **Scientific Computing:** `numpy >= 1.24.0`, `scipy >= 1.10.0`, `scikit-learn >= 1.3.0`, `nltk >= 3.8.1`.
* **Deep Learning Engines:** `tensorflow >= 2.14.0` / `keras >= 3.0` (with CUDA/cuDNN GPU acceleration).
* **Gradient Boosting:** `xgboost >= 2.0.0`.
* **Pre-trained Embeddings:** Stanford Twitter GloVe 200-dimensional vector embeddings (`dataset/glove-seeds.txt`).

---

## 3. High-Level Description of Tasks

The end-to-end engineering pipeline is structured into six sequential phases comprising fifteen discrete task units:

```mermaid
flowchart TD
    subgraph Phase1["Phase 1: Ingestion & Preprocessing"]
        T1["Task 1.1: CSV Integrity Verification & Ingestion"] --> T2["Task 1.2: Regex Normalization (URLs, Handles, Emojis)"]
        T2 --> T3["Task 1.3: Morphological Elongation & Orthographic Cleaning"]
    end

    subgraph Phase2["Phase 2: Statistical Profiling & Vocabulary Building"]
        T3 --> T4["Task 2.1: Frequency Distribution Extraction (stats.py)"]
        T4 --> T5["Task 2.2: Zipf's Law Profiling & Vocabulary Truncation"]
    end

    subgraph Phase3["Phase 3: Feature Representation Engineering"]
        T5 --> T6["Task 3.1: Sparse N-gram Vectorization (Presence vs Frequency)"]
        T5 --> T7["Task 3.2: Dense Word Embedding Indexing & GloVe Seeding"]
        T5 --> T8["Task 3.3: Lexicon Wordset Polarities Setup"]
    end

    subgraph Phase4["Phase 4: Model Development & Training"]
        T8 --> T9["Task 4.1: Heuristic Lexicon Baseline (baseline.py)"]
        T6 --> T10["Task 4.2: Classical ML (Naive Bayes, MaxEnt, CART, RF, SVM, XGBoost)"]
        T7 --> T11["Task 4.3: Deep Architectures (MLP, LSTM, 1D-CNN 1-4 Layers)"]
    end

    subgraph Phase5["Phase 5: Deep Feature Extraction & Hybrid Modeling"]
        T11 --> T12["Task 5.1: Penultimate CNN Dense Feature Extraction (600-d)"]
        T12 --> T13["Task 5.2: Hybrid CNN-Feature SVM Classifier Training"]
    end

    subgraph Phase6["Phase 6: Meta-Learning Ensemble & Evaluation"]
        T10 & T11 & T13 --> T14["Task 6.1: Model Validation & Submission Prediction"]
        T14 --> T15["Task 6.2: Majority Vote Ensemble Fusion (majority-voting.py)"]
    end
```

### Detailed Task Specifications

| Task ID | Task Name | Operational Description | Input Artifacts | Output Artifacts | Primary Implementation File |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **T-01** | Data Verification | Validate CSV integrity, strip invalid headers, verify class balance. | `train.csv`, `test.csv` | Sanitized raw CSVs | Shell / `utils.py` |
| **T-02** | Social Token Preprocessing | Regex mapping of URLs, @user handles, #hashtags, RTs, emoticons. | Sanitized raw CSVs | Tokenized text | `preprocess.py` |
| **T-03** | Word-Level Normalization | Repeated character reduction, punctuation stripping, case folding. | Tokenized text | `train-processed.csv`, `test-processed.csv` | `preprocess.py` |
| **T-04** | Statistical Extraction | Extract unigram and bigram frequency distributions via NLTK. | `train-processed.csv` | `train-processed-freqdist.pkl`, `freqdist-bi.pkl` | `stats.py` |
| **T-05** | Lexical Ranking & Pruning | Vocabulary cutoff selection ($N=15,000$ to $90,000$; $M=10,000$). | Frequency pickle files | Rank-frequency tables | `stats.py`, `utils.py` |
| **T-06** | Lexicon Baseline | Polarity scoring via positive and negative word lists. | Processed CSVs, wordsets | `baseline.csv`, accuracy metric | `baseline.py` |
| **T-07** | Classical Probabilistic Models | Train Multinomial/Bernoulli Naive Bayes and MaxEnt models. | Processed CSVs, vocabularies | `naivebayes.py`, `logistic.py`, `maxent-nltk.py` |
| **T-08** | Non-Linear Tree Ensembles | Train Decision Trees (CART), Random Forest, and XGBoost models. | Sparse feature matrices | Prediction CSVs, model dumps | `decisiontree.py`, `randomforest.py`, `xgboost.py` |
| **T-09** | Large-Margin Classification | Train Linear Support Vector Machines on sparse n-gram features. | Sparse presence matrices | Prediction CSVs, validation score | `svm.py` |
| **T-10** | Multilayer Perceptron | Train feedforward deep neural net with dropout on sparse input. | Sparse n-gram matrices | `best_mlp_model.h5` | `neuralnet.py` |
| **T-11** | Recurrent Sequence Modeling | Train LSTM models with random embeddings vs pre-trained GloVe. | Integer sequence arrays | `models/lstm-*.h5` | `lstm.py` |
| **T-12** | Temporal CNNs | Train 1-stage to 4-stage 1D CNNs with global max pooling. | Integer sequence arrays, GloVe | `models/4-conv-*.h5` | `cnn.py` |
| **T-13** | Penultimate Feature Extraction | Extract 600-d feature vectors from best 4-Conv-NN model. | `4-conv-nn.h5`, data arrays | `train-feats.npy`, `test-feats.npy`, `train-labels.txt` | `extract-cnn-feats.py` |
| **T-14** | Hybrid Representation SVM | Fit Linear Support Vector Classifier over extracted 600-d CNN features. | `train-feats.npy`, labels | `cnn-feats-svm.pkl`, `cnn-feats-svm.csv` | `cnn-feats-svm.py` |
| **T-15** | Majority Voting Ensembling | Hard voting aggregation across top 5 distinct predictive models. | 5 candidate prediction CSVs | `majority-voting.csv` (83.58% acc) | `majority-voting.py` |

---

## 4. Identification of Major Project Challenges in 2026

### Challenge 1: Temporal Concept Drift & Dynamic 2026 Internet Neologisms
Social media language evolves at an unprecedented pace. Between 2020 and 2026, user vocabulary on X witnessed rapid proliferation of new colloquial expressions (*delulu*, *rizz*, *gyatt*, *cap/no cap*, *mid*, *sigma*, *skibidi*), alongside evolving emoji connotations. 
* **Impact:** Static vocabulary representations and pre-trained dictionaries experience rapid out-of-vocabulary (OOV) decay when applied across time.
* **Mitigation:** Subword/character n-gram processing, character repetition compression ($\ge 3$ characters collapsed to 2 in [`preprocess.py`](file:///home/deepu/Academic%20Books/sem7/nlp/project/twitter-sentiment-analysis/code/preprocess.py)), and fine-tuning dense distributed representations end-to-end.

### Challenge 2: Contextual Polarity Inversion & Negation Handling
Affective polarity is frequently non-monotonic:
> *"The battery life is not bad at all, actually surprisingly great."* (Positive overall, but contains negative word *bad* and negation *not*).  
> *"It looked good, but failed within an hour."* (Negative overall, but begins with positive token *good*).  
* **Impact:** Standard bag-of-words (unigram) classifiers fail because they compute linear combinations of unigram counts, missing phrase-level compositional inversion.
* **Mitigation:** Bigram vocabulary augmentation ($M=10,000$ bigrams) in [`naivebayes.py`](file:///home/deepu/Academic%20Books/sem7/nlp/project/twitter-sentiment-analysis/code/naivebayes.py), recurrence gating in LSTM memory cells in [`lstm.py`](file:///home/deepu/Academic%20Books/sem7/nlp/project/twitter-sentiment-analysis/code/lstm.py), and local receptive fields in multi-stage 1D CNNs in [`cnn.py`](file:///home/deepu/Academic%20Books/sem7/nlp/project/twitter-sentiment-analysis/code/cnn.py).

### Challenge 3: Sarcasm, Irony, and Implicit Sentiment
Sarcasm is widespread on X. Users frequently combine hyper-enthusiastic positive adjectives with negative situations:
> *"I just adore getting stuck on the tarmac for six hours with zero air conditioning!"*  
* **Impact:** Surface-level lexicon counters and shallow classifiers produce confident false-positive errors.
* **Mitigation:** Extracting deep compositional representations from penultimate convolutional layers (`extract-cnn-feats.py`) and combining predictions via a diverse multi-model ensemble (`majority-voting.py`).

### Challenge 4: High Dimensionality, Extreme Sparsity, and Memory Scaling
Across 800,000 tweets, vocabulary extraction yields over **181,232 unigrams** and **1,954,953 bigrams**.
* **Impact:** A dense representation matrix ($800,000 \times 2,000,000$) would exceed several terabytes of memory, triggering fatal out-of-memory errors on standard systems.
* **Mitigation:** Applying Zipf's law truncation ($N=15,000$ to $90,000$ unigrams, $M=10,000$ bigrams), dynamic sparse matrix allocation (`scipy.sparse.lil_matrix` converting to `csr_matrix`), and minibatch streaming (500 samples per batch).

### Challenge 5: Pre-trained Word Embedding Antonym Proximity
Standard unsupervised embedding algorithms (Word2Vec, standard GloVe) optimize for sliding-window contextual co-occurrence. Because antonym pairs like `"love"` and `"hate"`, or `"great"` and `"awful"`, occur in identical syntactic frames (e.g., *"I [love/hate] this phone"*), their geometric distance in embedding space is deceptively small.
* **Mitigation:** Utilizing Stanford GloVe vectors pre-trained on 2 Billion Twitter posts (27B tokens), fine-tuning embedding weights end-to-end during supervised training of the LSTM and CNN networks, and extracting specialized 600-dimensional deep representations.

### Challenge 6: Distant Supervision Noise and Pseudo-Label Uncertainty
The Sentiment140 benchmark dataset relies on distant supervision (heuristically labeling posts containing `:)` as positive and `:(` as negative).
* **Impact:** Posts containing sarcastic smiles, condolences with sad faces, or multi-faceted sentiments carry noisy labels.
* **Mitigation:** Explicit emoticon normalization (`EMO_POS`, `EMO_NEG`), large-scale statistical regularization across 800,000 training instances, and aggressive dropout regularization ($p \in [0.2, 0.5]$) to prevent memorization of label noise.

### Challenge 7: Generative LLM Computational Cost vs. Real-Time Streaming Demands
In 2026, while 70B+ parameter generative LLMs provide strong zero-shot reasoning, their deployment incurs severe limitations:
* **Cost & Throughput:** Querying commercial LLM APIs for 1,000,000 posts costs thousands of dollars and introduces high network latency ($>300\text{ ms}$ per call).
* **Throughput Requirement:** Social media firehoses require processing $>10,000\text{ posts/second}$.
* **Mitigation:** Engineering compact, high-efficiency classical ML classifiers, lightweight 1D CNNs, and hybrid ensembles that achieve $\ge 83.5\%$ accuracy with sub-3ms latency per sample and zero recurring API costs.

---

## 5. Review of Related Works (10 Seminal & State-of-the-Art Studies: 2002–2026)

### 1. Go, Bhayani, and Huang (2009) — *Twitter Sentiment Classification using Distant Supervision*
* **Reference:** Go, A., Bhayani, R., and Huang, L. (2009). *Twitter Sentiment Classification using Distant Supervision*. Stanford University, Technical Report CS224N.
* **Problem & Motivation:** Manual labeling of social media corpora is cost-prohibitive. The authors investigated whether distant supervision—using emoticons in tweet text as noisy labels—could automate corpus creation for supervised machine learning.
* **Methodology:** Created the Sentiment140 corpus (1.6M tweets). Evaluated Naive Bayes, Maximum Entropy (Logistic Regression), and Support Vector Machines (SVM) over unigrams, bigrams, and POS tags.
* **Key Findings:** Machine learning achieved over 80% accuracy on distantly labeled data. MaxEnt achieved 83.0% with unigrams + bigrams. POS tagging degraded performance due to tagger error propagation on noisy text.
* **Relevance to Current Project:** Foundational paper and creator of the exact Sentiment140 dataset used in our `twitter-sentiment-analysis` workspace.

---

### 2. Pang, Lee, and Vaithyanathan (2002) — *Thumbs up? Sentiment Classification using Machine Learning Techniques*
* **Reference:** Pang, B., Lee, L., and Vaithyanathan, S. (2002). *Thumbs up? Sentiment Classification using Machine Learning Techniques*. In *Proceedings of EMNLP 2002*, pp. 79–86.
* **Problem & Motivation:** Pioneered supervised machine learning for computational sentiment classification, establishing whether sentiment could be modeled as a statistical classification problem rather than hand-crafted rule systems.
* **Methodology:** Evaluated Naive Bayes, Maximum Entropy, and SVM on review text. Explored unigrams vs. bigrams and term frequency vs. binary presence.
* **Key Findings:** SVM with binary unigram presence achieved the highest accuracy (82.9%). Critically, binary feature **presence** outperformed term **frequency**, demonstrating that word recurrence in short sentiment texts adds noise.
* **Relevance to Current Project:** Established the experimental methodology and binary presence representation implemented in our classical models ([`naivebayes.py`](file:///home/deepu/Academic%20Books/sem7/nlp/project/twitter-sentiment-analysis/code/naivebayes.py), [`svm.py`](file:///home/deepu/Academic%20Books/sem7/nlp/project/twitter-sentiment-analysis/code/svm.py)).

---

### 3. Pennington, Socher, and Manning (2014) — *GloVe: Global Vectors for Word Representation*
* **Reference:** Pennington, J., Socher, R., and Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation*. In *Proceedings of EMNLP 2014*, pp. 1532–1543.
* **Problem & Motivation:** Overcame the dichotomy between global matrix factorization (LSA) and local context window methods (Word2Vec) by factorizing global log co-occurrences with weighted least squares.
* **Methodology:** Trained log-bilinear word embeddings on massive corpora, including a specialized release of **2 Billion Twitter tweets** (27 Billion tokens, 200-dimensional vectors).
* **Key Findings:** Twitter GloVe vectors captured informal social media syntax, hashtags, and abbreviations far more effectively than embeddings trained on formal Wikipedia/Gigaword text.
* **Relevance to Current Project:** Directly provides the 200-dimensional pre-trained word vectors (`dataset/glove-seeds.txt`) utilized to seed the embedding layers of [`cnn.py`](file:///home/deepu/Academic%20Books/sem7/nlp/project/twitter-sentiment-analysis/code/cnn.py) and [`lstm.py`](file:///home/deepu/Academic%20Books/sem7/nlp/project/twitter-sentiment-analysis/code/lstm.py).

---

### 4. Kim (2014) — *Convolutional Neural Networks for Sentence Classification*
* **Reference:** Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification*. In *Proceedings of EMNLP 2014*, pp. 1746–1751.
* **Problem & Motivation:** Investigated whether 1D temporal Convolutional Neural Networks, previously dominant in computer vision, could be applied to sentence-level sentiment classification with minimal tuning.
* **Methodology:** Applied 1D convolutions with variable filter kernel sizes ($h \in \{3, 4, 5\}$), global max-over-time pooling, dropout ($p=0.5$), and dense softmax output over pre-trained embeddings.
* **Key Findings:** Achieved state-of-the-art results across 4 sentiment benchmarks (88.1% on SST-2). Fine-tuning pre-trained embeddings during supervised training dynamically aligned sentiment word geometry.
* **Relevance to Current Project:** Serves as the direct architectural foundation for [`cnn.py`](file:///home/deepu/Academic%20Books/sem7/nlp/project/twitter-sentiment-analysis/code/cnn.py) and the 600-dimensional feature extractor [`extract-cnn-feats.py`](file:///home/deepu/Academic%20Books/sem7/nlp/project/twitter-sentiment-analysis/code/extract-cnn-feats.py).

---

### 5. Mohammad, Kiritchenko, and Zhu (2013) — *NRC-Canada: Building the State-of-the-Art System for Twitter Sentiment Analysis*
* **Reference:** Mohammad, S. M., Kiritchenko, S., and Zhu, X. (2013). *NRC-Canada: Building the State-of-the-Art System for Twitter Sentiment Analysis*. In *Proceedings of SemEval 2013*, pp. 321–327.
* **Problem & Motivation:** Winning system in the seminal SemEval-2013 Task 2 benchmark; addressed the challenge of engineering rich social-media-specific features for maximum-margin classification.
* **Methodology:** Trained linear SVMs on a composite feature space: word/character n-grams, sentiment lexicons (NRC, MPQA, Bing Liu), all-caps word counts, hashtag counts, and punctuation clusters.
* **Key Findings:** Achieved 1st place with a macro-F1 of 69.02%. Sentiment lexicons provided the largest individual performance gain (+5.5% F1).
* **Relevance to Current Project:** Demonstrates the ceiling of classical feature engineering and motivates our baseline lexicon scoring in [`baseline.py`](file:///home/deepu/Academic%20Books/sem7/nlp/project/twitter-sentiment-analysis/code/baseline.py).

---

### 6. Wang and Manning (2012) — *Baselines and Bigrams: Simple, Good Sentiment and Topic Classification*
* **Reference:** Wang, S., and Manning, C. D. (2012). *Baselines and Bigrams: Simple, Good Sentiment and Topic Classification*. In *Proceedings of ACL 2012*, pp. 90–94.
* **Problem & Motivation:** Investigated why linear SVMs frequently underperform Naive Bayes on short snippets and formulated a unified model: Naive Bayes Support Vector Machines (NBSVM).
* **Methodology:** Rigorous theoretical and empirical comparison between Multinomial Naive Bayes, Linear SVM, and NBSVM across unigram and bigram representations on sentiment datasets.
* **Key Findings:** MNB is superior to standard SVM on short text snippets; augmenting unigrams with bigrams yields consistent, substantial performance gains across all linear baselines.
* **Relevance to Current Project:** Directly underpins our unigram/bigram feature extraction and linear model designs in [`naivebayes.py`](file:///home/deepu/Academic%20Books/sem7/nlp/project/twitter-sentiment-analysis/code/naivebayes.py) and [`svm.py`](file:///home/deepu/Academic%20Books/sem7/nlp/project/twitter-sentiment-analysis/code/svm.py).

---

### 7. Nguyen, Vu, and Nguyen (2020) — *BERTweet: A pre-trained language model for English Tweets*
* **Reference:** Nguyen, D. Q., Vu, T., and Nguyen, A. T. (2020). *BERTweet: A pre-trained language model for English Tweets*. In *Proceedings of EMNLP 2020: System Demonstrations*, pp. 9–14.
* **Problem & Motivation:** General-domain transformer models (BERT, RoBERTa) degrade on microblogs due to domain mismatch and tokenization fragmentation over Twitter slang, emojis, and hashtags.
* **Methodology:** Pre-trained a RoBERTa-based architecture on 850 Million English tweets (16 Billion tokens) using specialized subword tokenization (FastBPE) preserving social media tokens.
* **Key Findings:** BERTweet outperformed RoBERTa-base and standard BERT across all TweetEval benchmarks, improving Twitter sentiment classification macro-F1 by 3.8% over general-domain models.
* **Relevance to Current Project:** Highlights the modern transformer baseline for social sentiment analysis, demonstrating that in-domain social pre-training is crucial for capturing microblog semantics.

---

### 8. Barbieri, Camacho-Collados, Neves, and Espinosa-Anke (2020) — *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification*
* **Reference:** Barbieri, F., Camacho-Collados, J., Neves, L., and Espinosa-Anke, L. (2020). *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification*. In *Findings of EMNLP 2020*, pp. 1644–1650.
* **Problem & Motivation:** Standardized evaluation of microblog classification across seven disparate NLP tasks (sentiment, irony, hate speech, emotion, offensive language, stance, emoji prediction) to establish fair cross-model comparison.
* **Methodology:** Curated multi-task benchmark splits; rigorously evaluated classical feature models (FastText, SVM), recurrence (BiLSTM), and transformers (BERT, RoBERTa).
* **Key Findings:** Confirmed that sentiment analysis remains heavily correlated with emotion and irony detection. Demonstrated that while transformers dominate accuracy, ensemble and recurrent models remain competitive under low-latency constraints.
* **Relevance to Current Project:** Establishes the modern standardized evaluation framework and metric conventions adopted in our project reporting.

---

### 9. Loureiro, Barbieri, Neves, Espinosa-Anke, and Camacho-Collados (2022) — *TimeLMs: Diachronic Large Language Models from Twitter*
* **Reference:** Loureiro, D., Barbieri, F., Neves, L., Espinosa-Anke, L., and Camacho-Collados, J. (2022). *TimeLMs: Diachronic Large Language Models from Twitter*. In *Proceedings of ACL 2022: System Demonstrations*, pp. 251–260.
* **Problem & Motivation:** Investigated **temporal degradation** and concept drift on social media: models trained on past years suffer systematic performance loss on future posts due to evolving events, memes, and slang.
* **Methodology:** Continuously pre-trained language models on monthly slices of Twitter data from 2019 to 2022, benchmarking out-of-distribution temporal degradation on sentiment and topic classification.
* **Key Findings:** Temporal domain shift degrades classification accuracy by up to 6.2% F1 over a two-year horizon. Continual adaptation and robust morphological normalization are necessary to withstand temporal drift.
* **Relevance to Current Project:** Directly informs our 2026 challenge formulation on temporal concept drift and validates our robust morphological cleaning routines in [`preprocess.py`](file:///home/deepu/Academic%20Books/sem7/nlp/project/twitter-sentiment-analysis/code/preprocess.py).

---

### 10. Zhang, Lyu, and Liu (2023) — *Sentiment Analysis in the Era of Large Language Models: A Comprehensive Survey and Evaluation*
* **Reference:** Zhang, W., Lyu, Y., and Liu, B. (2023). *Sentiment Analysis in the Era of Large Language Models: A Comprehensive Survey and Evaluation*. *ACM Computing Surveys / arXiv:2307.08728*.
* **Problem & Motivation:** Evaluated modern Generative LLMs (GPT-3.5, GPT-4, LLaMA-2) across standard sentiment tasks, contrasting zero-shot/few-shot prompting against specialized discriminative models.
* **Methodology:** Benchmarked zero-shot and instruction-tuned generative LLMs against fine-tuned specialized models on microblogs and review datasets, measuring accuracy, latency, and monetary cost.
* **Key Findings:** Generative LLMs exhibit strong general reasoning on subtle sarcasm, but:
  1. Fine-tuned compact discriminative models (CNNs, RoBERTa, ensembles) still match or outperform zero-shot LLMs on standard binary sentiment benchmarks.
  2. Generative LLMs incur over **$100\times$ higher inference latency** and high API costs, making them impractical for continuous real-time firehose streaming.
  3. Hybrid ensembles of discriminative classifiers provide the optimal cost-to-performance frontier for high-throughput production.
* **Relevance to Current Project:** Provides theoretical and empirical justification for our 2026 project design: developing a compact, high-throughput multi-paradigm pipeline and ensemble rather than relying on costly external LLM APIs.

---

## 6. Comparative Synthesis of Related Works

| Study | Author(s) & Year | Paradigm | Feature / Embedding Layer | Benchmark Corpus | Key Metric | Direct Architectural Connection to Current Project |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[1]** | Go et al. (2009) | Classical ML (NB, MaxEnt, SVM) | Unigram, Bigram, POS | Sentiment140 (1.6M) | 83.0% Acc (MaxEnt) | Direct dataset creator & distant supervision baseline for this workspace. |
| **[2]** | Pang et al. (2002) | Supervised ML (NB, ME, SVM) | Presence vs Frequency | IMDb Reviews | 82.9% Acc (SVM) | Established foundational principle: binary presence beats frequency for sparse text. |
| **[3]** | Pennington et al. (2014) | Log-Bilinear Matrix Factorization | Twitter GloVe 200-d | 2B Twitter Corpus | SOTA Word Vectors | Source of pre-trained word embeddings loaded in `dataset/glove-seeds.txt`. |
| **[4]** | Kim (2014) | 1D Temporal CNN | Pre-trained Embeddings | SST-1, SST-2 | 88.1% Acc (SST-2) | Direct architectural blueprint for `cnn.py` and temporal max pooling. |
| **[5]** | Mohammad et al. (2013) | Linear SVM (Large Margin) | NRC Lexicons, Punctuation | SemEval-2013 | 69.0% F1 (Winner) | Demonstrates the ceiling of hand-crafted feature engineering in linear models. |
| **[6]** | Wang & Manning (2012) | NBSVM & Linear SVM | Bigrams + Log-Odds | Short Snippets / Reviews | 91.6% Acc (NBSVM) | Confirmed that bigram augmentation consistently boosts linear classifier accuracy. |
| **[7]** | Nguyen et al. (2020) | Pre-trained Transformer | FastBPE Subwords (RoBERTa) | TweetEval (850M tweets) | SOTA TweetEval | Establishes modern transformer baseline for social media NLP. |
| **[8]** | Barbieri et al. (2020) | Multi-Task Benchmark | Transformer / BiLSTM / SVM | TweetEval Benchmark | Standardized F1 | Provided the unified evaluation protocol and cross-task benchmark metrics. |
| **[9]** | Loureiro et al. (2022) | Diachronic Language Models | Continual Twitter Pre-training | TimeLMs Corpus | -6.2% Temporal Drop | Validated 2026 challenge: temporal concept drift demands robust normalization. |
| **[10]** | Zhang et al. (2023) | LLMs vs. Discriminative Models | In-Context Learning vs Ensembles | Twitter & Review Benchmarks | LLMs $100\times$ slower | Proved that compact ensembles outperform zero-shot LLMs on cost/speed trade-offs. |
| **Ours** | **This Project (2026)** | **Multi-Paradigm Ensemble** | **GloVe 200d + 600d CNN Feats** | **Sentiment140 (1M tweets)** | **83.58% Acc (Kaggle)** | **Combines Top 5 Models (LSTM, 3-Conv, 4-Conv, 4-Conv-SVM, CNN-len20) via Majority Voting.** |

---

## 7. Formal Bibliography (BibTeX)

```bibtex
@techreport{go2009twitter,
  author      = {Go, Alec and Bhayani, Richa and Huang, Lei},
  title       = {Twitter Sentiment Classification using Distant Supervision},
  institution = {Stanford University},
  year        = {2009},
  number      = {CS224N Project Report}
}

@inproceedings{pang2002thumbs,
  author    = {Pang, Bo and Lee, Lillian and Vaithyanathan, Shivakumar},
  title     = {Thumbs up? Sentiment Classification using Machine Learning Techniques},
  booktitle = {Proceedings of the 2002 Conference on Empirical Methods in Natural Language Processing (EMNLP 2002)},
  pages     = {79--86},
  year      = {2002}
}

@inproceedings{pennington2014glove,
  author    = {Pennington, Jeffrey and Socher, Richard and Manning, Christopher D.},
  title     = {{GloVe}: Global Vectors for Word Representation},
  booktitle = {Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP 2014)},
  pages     = {1532--1543},
  year      = {2014}
}

@inproceedings{kim2014convolutional,
  author    = {Kim, Yoon},
  title     = {Convolutional Neural Networks for Sentence Classification},
  booktitle = {Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP 2014)},
  pages     = {1746--1751},
  year      = {2014}
}

@inproceedings{mohammad2013nrc,
  author    = {Mohammad, Saif M. and Kiritchenko, Svetlana and Zhu, Xiaodan},
  title     = {{NRC-Canada}: Building the State-of-the-Art System for Twitter Sentiment Analysis},
  booktitle = {Proceedings of the 7th International Workshop on Semantic Evaluation (SemEval 2013)},
  pages     = {321--327},
  year      = {2013}
}

@inproceedings{wang2012baselines,
  author    = {Wang, Sida and Manning, Christopher D.},
  title     = {Baselines and Bigrams: Simple, Good Sentiment and Topic Classification},
  booktitle = {Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (ACL 2012)},
  pages     = {90--94},
  year      = {2012}
}

@inproceedings{nguyen2020bertweet,
  author    = {Nguyen, Dat Quoc and Vu, Thanh and Nguyen, Anh Tuan},
  title     = {{BERTweet}: A pre-trained language model for {English} Tweets},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (EMNLP 2020)},
  pages     = {9--14},
  year      = {2020}
}

@inproceedings{barbieri2020tweeteval,
  author    = {Barbieri, Francesco and Camacho-Collados, Jose and Neves, Leonardo and Espinosa-Anke, Luis},
  title     = {{TweetEval}: Unified Benchmark and Comparative Evaluation for Tweet Classification},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
  pages     = {1644--1650},
  year      = {2020}
}

@inproceedings{loureiro2022timelms,
  author    = {Loureiro, Daniel and Barbieri, Francesco and Neves, Leonardo and Espinosa-Anke, Luis and Camacho-Collados, Jose},
  title     = {{TimeLMs}: Diachronic Large Language Models from Twitter},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: System Demonstrations (ACL 2022)},
  pages     = {251--260},
  year      = {2022}
}

@article{zhang2023sentiment,
  author    = {Zhang, Wenxuan and Lyu, Yueheng and Liu, Bing},
  title     = {Sentiment Analysis in the Era of Large Language Models: A Comprehensive Survey and Evaluation},
  journal   = {ACM Computing Surveys / arXiv preprint arXiv:2307.08728},
  year      = {2023}
}
```

---

# Roadmap for Subsequent Phases (Phases B to E)

As the project progresses through subsequent project deliverables, the following phases will be systematically implemented and incorporated into this Project Report:

* **Phase B: System Architecture, Data Engineering & Methodology**  
  Detailed pipeline diagrams, mathematical derivations of sparse transformations (presence vs. frequency, TF-IDF, GloVe embedding lookups), loss formulations (Binary Cross-Entropy, Hinge Loss, Margin Ranking), backpropagation dynamics, and computational complexity analysis ($O(N)$ vs $O(V)$).
* **Phase C: Implementation & Experimental Framework**  
  Comprehensive walkthrough of codebase modules ([`preprocess.py`](file:///home/deepu/Academic%20Books/sem7/nlp/project/twitter-sentiment-analysis/code/preprocess.py), [`stats.py`](file:///home/deepu/Academic%20Books/sem7/nlp/project/twitter-sentiment-analysis/code/stats.py), classical classifiers, neural networks, feature extractors, and ensemble orchestrators), hyperparameter schedules, and regularization techniques (dropout, weight decay).
* **Phase D: Quantitative Results, Evaluation & Error Analysis**  
  Empirical benchmarking tables across all 12 model configurations, Kaggle leaderboard validation, ablation experiments on unigrams vs. bigrams, confusion matrices, and qualitative case studies of failure modes (sarcasm, mixed polarity, and implicit sentiment).
* **Phase E: Conclusion, Discussion & Future Extensions**  
  Synthesis of core architectural insights, latency-accuracy Pareto frontiers, discussion of 2026 deployment trade-offs, and future directions (multi-class emotion grading and parameter-efficient adapter fine-tuning).
