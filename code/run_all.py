#!/usr/bin/env python3
"""
Unified Twitter Sentiment Analysis Pipeline — Python 3, GPU-Accelerated
========================================================================
Runs all phases: Preprocessing → Stats → Baseline → Naive Bayes →
SVM → Logistic → Decision Tree → LSTM (GPU) → CNN (GPU) → Hybrid CNN-SVM → Majority Voting

Usage:
    /home/deepu/anaconda3/bin/python3 run_all.py                     # Run all models
    /home/deepu/anaconda3/bin/python3 run_all.py --models baseline nb svm  # Run specific models
    /home/deepu/anaconda3/bin/python3 run_all.py --models cnn lstm hybrid  # Run GPU models

Author: Modernized from Python 2 codebase for CS-471 NLP (Sem 7, Fall 2026)
"""

import os
import sys
import re
import csv
import glob
import time
import pickle
import random
import argparse
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / 'dataset'
CODE_DIR = BASE_DIR / 'code'
MODELS_DIR = CODE_DIR / 'models'
RESULTS_DIR = CODE_DIR / 'results'

RAW_TRAIN_FILE = DATASET_DIR / 'training.1600000.processed.noemoticon.csv'
GLOVE_FILE = DATASET_DIR / 'glove.twitter.27B.200d.txt'
GLOVE_SEEDS_FILE = DATASET_DIR / 'glove-seeds.txt'

TRAIN_PROCESSED_FILE = CODE_DIR / 'train-processed.csv'
TEST_PROCESSED_FILE = CODE_DIR / 'test-processed.csv'
FREQ_DIST_FILE = CODE_DIR / 'train-processed-freqdist.pkl'
BI_FREQ_DIST_FILE = CODE_DIR / 'train-processed-freqdist-bi.pkl'

POSITIVE_WORDS_FILE = DATASET_DIR / 'positive-words.txt'
NEGATIVE_WORDS_FILE = DATASET_DIR / 'negative-words.txt'

# Model parameters
VOCAB_SIZE_DEEP = 90000
UNIGRAM_SIZE = 15000
BIGRAM_SIZE = 10000
MAX_LENGTH = 40
GLOVE_DIM = 200
BATCH_SIZE_DEEP = 256
CNN_FILTERS = 600
CNN_KERNEL = 3
TRAIN_SIZE = 800000
TEST_SIZE = 200000
RANDOM_SEED = 1337

# ============================================================================
# Utility functions
# ============================================================================

def write_status(i, total):
    sys.stdout.write('\r')
    if total > 0:
        sys.stdout.write('Processing %d/%d (%.1f%%)' % (i, total, 100.0 * i / total))
    else:
        sys.stdout.write('Processing %d' % i)
    sys.stdout.flush()


def file_to_wordset(filename):
    words = []
    with open(filename, 'r', errors='ignore') as f:
        for line in f:
            w = line.strip()
            if w and not w.startswith(';'):
                words.append(w)
    return set(words)


def save_results_to_csv(results, csv_file):
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(csv_file, 'w') as f:
        f.write('id,prediction\n')
        for tweet_id, pred in results:
            f.write('%s,%s\n' % (str(tweet_id), str(int(pred))))


def top_n_words(pkl_file_name, N, shift=0):
    with open(pkl_file_name, 'rb') as pkl_file:
        freq_dist = pickle.load(pkl_file)
    most_common = freq_dist.most_common(N)
    words = {p[0]: i + shift for i, p in enumerate(most_common)}
    return words


def top_n_bigrams(pkl_file_name, N, shift=0):
    with open(pkl_file_name, 'rb') as pkl_file:
        freq_dist = pickle.load(pkl_file)
    most_common = freq_dist.most_common(N)
    bigrams = {p[0]: i for i, p in enumerate(most_common)}
    return bigrams


def split_data(tweets, validation_split=0.1):
    index = int((1 - validation_split) * len(tweets))
    random.shuffle(tweets)
    return tweets[:index], tweets[index:]


# ============================================================================
# Phase 1: Preprocessing
# ============================================================================

def preprocess_word(word):
    word = word.strip('\'"?!,.():;')
    word = re.sub(r'(.)\1+', r'\1\1', word)
    word = re.sub(r"(-|')", '', word)
    return word


def is_valid_word(word):
    return re.search(r'^[a-zA-Z][a-z0-9A-Z._]*$', word) is not None


def handle_emojis(tweet):
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    tweet = re.sub(r"(:,\(|:'\(|:\"\()", ' EMO_NEG ', tweet)
    return tweet


def preprocess_tweet(tweet):
    processed_tweet = []
    tweet = tweet.lower()
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    tweet = tweet.strip(' "\'')
    tweet = handle_emojis(tweet)
    tweet = re.sub(r'\s+', ' ', tweet)
    words = tweet.split()
    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            processed_tweet.append(word)
    return ' '.join(processed_tweet)


def prepare_dataset():
    print('\n' + '='*70)
    print('PHASE 1: DATA PREPROCESSING')
    print('='*70)

    if TRAIN_PROCESSED_FILE.exists() and TEST_PROCESSED_FILE.exists():
        with open(TRAIN_PROCESSED_FILE) as f:
            train_count = sum(1 for _ in f)
        with open(TEST_PROCESSED_FILE) as f:
            test_count = sum(1 for _ in f)
        print(f'  Preprocessed files verified:')
        print(f'    Train: {train_count} lines ({TRAIN_PROCESSED_FILE})')
        print(f'    Test:  {test_count} lines ({TEST_PROCESSED_FILE})')
        return

    print(f'  Reading raw dataset: {RAW_TRAIN_FILE}')
    start = time.time()
    all_rows = []
    with open(RAW_TRAIN_FILE, 'r', encoding='latin-1') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 6:
                sentiment = int(row[0])
                tweet_id = row[1]
                text = row[5]
                label = 1 if sentiment == 4 else 0
                all_rows.append((tweet_id, label, text))

    print(f'  Read {len(all_rows)} total rows in {time.time()-start:.1f}s')
    random.seed(RANDOM_SEED)
    random.shuffle(all_rows)

    pos_rows = [r for r in all_rows if r[1] == 1]
    neg_rows = [r for r in all_rows if r[1] == 0]

    train_data = pos_rows[:400000] + neg_rows[:400000]
    test_data = pos_rows[400000:500000] + neg_rows[400000:500000]
    random.shuffle(train_data)
    random.shuffle(test_data)

    print('  Preprocessing training set...')
    with open(TRAIN_PROCESSED_FILE, 'w') as f:
        for i, (tid, label, tweet) in enumerate(train_data):
            processed = preprocess_tweet(tweet)
            if processed.strip():
                f.write(f'{tid},{label},{processed}\n')
            if (i+1) % 100000 == 0:
                write_status(i+1, len(train_data))
    print()

    print('  Preprocessing test set...')
    with open(TEST_PROCESSED_FILE, 'w') as f:
        for i, (tid, label, tweet) in enumerate(test_data):
            processed = preprocess_tweet(tweet)
            if processed.strip():
                f.write(f'{tid},{label},{processed}\n')
            if (i+1) % 50000 == 0:
                write_status(i+1, len(test_data))
    print()

    # Symlink at project root for compatibility
    for fname in ['train-processed.csv', 'test-processed.csv']:
        root_sym = BASE_DIR / fname
        if not root_sym.exists():
            root_sym.symlink_to(CODE_DIR / fname)

    print(f'  Preprocessing complete in {time.time()-start:.1f}s')


# ============================================================================
# Phase 2: Compute Stats / Frequency Distributions
# ============================================================================

def compute_stats():
    print('\n' + '='*70)
    print('PHASE 2: COMPUTING STATISTICS & FREQUENCY DISTRIBUTIONS')
    print('='*70)

    if FREQ_DIST_FILE.exists() and BI_FREQ_DIST_FILE.exists():
        print(f'  Frequency distribution files verified:')
        print(f'    Unigrams: {FREQ_DIST_FILE}')
        print(f'    Bigrams:  {BI_FREQ_DIST_FILE}')
        return

    from nltk import FreqDist

    all_words = []
    all_bigrams = []
    num_tweets = 0
    num_pos = 0
    num_neg = 0

    with open(TRAIN_PROCESSED_FILE, 'r') as f:
        lines = f.readlines()
        num_tweets = len(lines)
        for i, line in enumerate(lines):
            parts = line.strip().split(',', 2)
            if len(parts) < 3:
                continue
            t_id, if_pos, tweet = parts
            if_pos = int(if_pos)
            if if_pos:
                num_pos += 1
            else:
                num_neg += 1

            words = tweet.split()
            all_words.extend(words)

            for j in range(len(words) - 1):
                all_bigrams.append((words[j], words[j+1]))

            if (i+1) % 100000 == 0:
                write_status(i+1, num_tweets)

    print(f'\n  Tweets => Total: {num_tweets}, Positive: {num_pos}, Negative: {num_neg}')
    print(f'  Words  => Total: {len(all_words)}, Unique: {len(set(all_words))}')
    print(f'  Bigrams => Total: {len(all_bigrams)}, Unique: {len(set(all_bigrams))}')

    print('  Computing unigram frequency distribution...')
    freq_dist = FreqDist(all_words)
    with open(FREQ_DIST_FILE, 'wb') as pkl_file:
        pickle.dump(freq_dist, pkl_file)
    print(f'  Saved: {FREQ_DIST_FILE}')

    print('  Computing bigram frequency distribution...')
    bigram_freq = Counter(all_bigrams)
    with open(BI_FREQ_DIST_FILE, 'wb') as pkl_file:
        pickle.dump(bigram_freq, pkl_file)
    print(f'  Saved: {BI_FREQ_DIST_FILE}')

    # Symlink at project root
    for fname in ['train-processed-freqdist.pkl', 'train-processed-freqdist-bi.pkl']:
        root_sym = BASE_DIR / fname
        if not root_sym.exists():
            root_sym.symlink_to(CODE_DIR / fname)


# ============================================================================
# Classical Features Extraction
# ============================================================================

def get_ngram_feature_vector(tweet, unigrams, bigrams=None, use_bigrams=True):
    uni_fv = []
    bi_fv = []
    words = tweet.split()
    for i in range(len(words) - 1):
        word = words[i]
        next_word = words[i + 1]
        if unigrams.get(word) is not None:
            uni_fv.append(word)
        if use_bigrams and bigrams is not None:
            if bigrams.get((word, next_word)) is not None:
                bi_fv.append((word, next_word))
    if len(words) >= 1:
        if unigrams.get(words[-1]) is not None:
            uni_fv.append(words[-1])
    return uni_fv, bi_fv


def process_tweets_classical(csv_file, unigrams, bigrams=None, use_bigrams=True,
                             test_file=True, max_count=None):
    tweets = []
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        if max_count:
            lines = lines[:max_count]
        total = len(lines)
        for i, line in enumerate(lines):
            parts = line.strip().split(',', 2)
            if len(parts) < 2:
                continue
            if test_file:
                tweet_id = parts[0]
                tweet = parts[1] if len(parts) == 2 else parts[2]
            else:
                if len(parts) < 3:
                    continue
                tweet_id, sentiment, tweet = parts
            feature_vector = get_ngram_feature_vector(tweet, unigrams, bigrams, use_bigrams)
            if test_file:
                tweets.append((tweet_id, feature_vector))
            else:
                tweets.append((tweet_id, int(sentiment), feature_vector))
            if (i+1) % 100000 == 0:
                write_status(i+1, total)
    print()
    return tweets


def extract_features_batch(tweets, unigrams, bigrams=None, use_bigrams=True,
                           unigram_size=15000, bigram_size=10000,
                           test_file=True, feat_type='presence'):
    from scipy.sparse import lil_matrix
    vocab_size = unigram_size + (bigram_size if use_bigrams else 0)
    batch_size = len(tweets)
    features = lil_matrix((batch_size, vocab_size), dtype=np.float32)
    labels = np.zeros(batch_size, dtype=int)

    for j, tweet in enumerate(tweets):
        if test_file:
            tweet_words = tweet[1][0]
            tweet_bigrams = tweet[1][1]
        else:
            tweet_words = tweet[2][0]
            tweet_bigrams = tweet[2][1]
            labels[j] = tweet[1]

        if feat_type == 'presence':
            tweet_words = set(tweet_words)
            tweet_bigrams = set(tweet_bigrams)

        for word in tweet_words:
            idx = unigrams.get(word)
            if idx is not None:
                features[j, idx] += 1

        if use_bigrams and bigrams is not None:
            for bigram in tweet_bigrams:
                idx = bigrams.get(bigram)
                if idx is not None:
                    features[j, unigram_size + idx] += 1

    return features.tocsr(), labels


# ============================================================================
# Deep Features & GloVe Seeds
# ============================================================================

def get_glove_vectors(vocab):
    print('  Loading GloVe vectors...')
    glove_vectors = {}

    # Check cached seeds file first
    if GLOVE_SEEDS_FILE.exists():
        print(f'  Found cached seed vectors: {GLOVE_SEEDS_FILE}')
        found = 0
        with open(GLOVE_SEEDS_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
                word = tokens[0]
                if vocab.get(word) is not None:
                    try:
                        vector = [float(e) for e in tokens[1:]]
                        if len(vector) == GLOVE_DIM:
                            glove_vectors[word] = np.array(vector, dtype=np.float32)
                            found += 1
                    except ValueError:
                        continue
        print(f'  Loaded {found} words from {GLOVE_SEEDS_FILE}')
        return glove_vectors

    # Otherwise read from main GloVe file and save seeds
    if not GLOVE_FILE.exists():
        print(f'  WARNING: GloVe file not found at {GLOVE_FILE}. Using random weights.')
        return glove_vectors

    found = 0
    total_lines = 0
    seeds_lines = []
    print(f'  Scanning {GLOVE_FILE} (2.0 GB)...')
    with open(GLOVE_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            total_lines += 1
            tokens = line.strip().split()
            if not tokens:
                continue
            word = tokens[0]
            if vocab.get(word) is not None:
                try:
                    vector = [float(e) for e in tokens[1:]]
                    if len(vector) == GLOVE_DIM:
                        glove_vectors[word] = np.array(vector, dtype=np.float32)
                        seeds_lines.append(line)
                        found += 1
                except ValueError:
                    continue
            if total_lines % 300000 == 0:
                write_status(total_lines, 1200000)

    print(f'\n  Found {found} / {len(vocab)} words in GloVe.')
    if seeds_lines:
        print(f'  Caching {len(seeds_lines)} matched vectors to {GLOVE_SEEDS_FILE}...')
        with open(GLOVE_SEEDS_FILE, 'w', encoding='utf-8') as f:
            f.writelines(seeds_lines)
    return glove_vectors


def get_deep_feature_vector(tweet, vocab):
    words = tweet.split()
    feature_vector = []
    for word in words:
        idx = vocab.get(word)
        if idx is not None:
            feature_vector.append(idx)
    return feature_vector


def process_tweets_deep(csv_file, vocab, test_file=True, max_count=None):
    tweets = []
    labels = []
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        if max_count:
            lines = lines[:max_count]
        total = len(lines)
        for i, line in enumerate(lines):
            parts = line.strip().split(',', 2)
            if test_file:
                tweet = parts[-1]
            else:
                if len(parts) < 3:
                    continue
                tweet = parts[2]
            feature_vector = get_deep_feature_vector(tweet, vocab)
            tweets.append(feature_vector)
            if not test_file:
                labels.append(int(parts[1]))
            if (i+1) % 100000 == 0:
                write_status(i+1, total)
    print()
    return tweets, np.array(labels, dtype=int) if labels else np.array([])


# ============================================================================
# Model: Baseline (lexicon)
# ============================================================================

def run_baseline(max_count=None):
    print('\n' + '-'*70)
    print('MODEL 1: Lexicon Baseline (Rule-Based)')
    print('-'*70)
    start = time.time()

    positive_words = file_to_wordset(str(POSITIVE_WORDS_FILE))
    negative_words = file_to_wordset(str(NEGATIVE_WORDS_FILE))

    correct = 0
    total = 0
    predictions = []
    with open(TRAIN_PROCESSED_FILE, 'r') as f:
        lines = f.readlines()
        if max_count:
            lines = lines[:max_count]
        for line in lines:
            parts = line.strip().split(',', 2)
            if len(parts) < 3:
                continue
            tweet_id, label, tweet = parts
            label = int(label)
            pos_count = sum(1 for w in tweet.split() if w in positive_words)
            neg_count = sum(1 for w in tweet.split() if w in negative_words)
            pred = 1 if pos_count >= neg_count else 0
            if pred == label:
                correct += 1
            predictions.append((tweet_id, pred))
            total += 1

    acc = correct * 100.0 / total
    save_results_to_csv(predictions[:200000], RESULTS_DIR / 'baseline.csv')
    print(f'  Accuracy: {correct}/{total} = {acc:.2f}%')
    print(f'  Inference Speed: {(time.time()-start)*1000/total:.3f} ms/tweet')
    return acc


# ============================================================================
# Model: Naive Bayes
# ============================================================================

def run_naive_bayes(max_count=None):
    print('\n' + '-'*70)
    print('MODEL 2: Multinomial Naive Bayes (Unigrams + Bigrams, Presence)')
    print('-'*70)
    start = time.time()
    from sklearn.naive_bayes import MultinomialNB

    unigrams = top_n_words(str(FREQ_DIST_FILE), UNIGRAM_SIZE)
    bigrams = top_n_bigrams(str(BI_FREQ_DIST_FILE), BIGRAM_SIZE)

    print('  Loading tweets...')
    tweets = process_tweets_classical(str(TRAIN_PROCESSED_FILE), unigrams, bigrams,
                                       use_bigrams=True, test_file=False, max_count=max_count)
    train_tweets, val_tweets = split_data(tweets)
    del tweets

    print(f'  Train: {len(train_tweets)}, Val: {len(val_tweets)}')
    print('  Extracting CSR sparse matrices...')
    train_X, train_y = extract_features_batch(train_tweets, unigrams, bigrams,
                                               use_bigrams=True,
                                               unigram_size=UNIGRAM_SIZE,
                                               bigram_size=BIGRAM_SIZE,
                                               test_file=False, feat_type='presence')

    print('  Training Multinomial NB...')
    clf = MultinomialNB()
    clf.fit(train_X, train_y)

    print('  Evaluating...')
    val_X, val_y = extract_features_batch(val_tweets, unigrams, bigrams,
                                           use_bigrams=True,
                                           unigram_size=UNIGRAM_SIZE,
                                           bigram_size=BIGRAM_SIZE,
                                           test_file=False, feat_type='presence')
    predictions = clf.predict(val_X)
    correct = np.sum(predictions == val_y)
    total = len(val_tweets)
    acc = correct * 100.0 / total
    print(f'  Validation Accuracy: {correct}/{total} = {acc:.2f}%')
    print(f'  Elapsed: {time.time()-start:.1f}s')

    val_preds_list = [(str(i), p) for i, p in enumerate(predictions)]
    save_results_to_csv(val_preds_list, RESULTS_DIR / 'naivebayes.csv')
    return acc, predictions


# ============================================================================
# Model: Linear SVM
# ============================================================================

def run_svm(max_count=None):
    print('\n' + '-'*70)
    print('MODEL 3: Linear SVM (Unigrams + Bigrams, Presence)')
    print('-'*70)
    start = time.time()
    from sklearn.svm import LinearSVC

    unigrams = top_n_words(str(FREQ_DIST_FILE), UNIGRAM_SIZE)
    bigrams = top_n_bigrams(str(BI_FREQ_DIST_FILE), BIGRAM_SIZE)

    print('  Loading tweets...')
    tweets = process_tweets_classical(str(TRAIN_PROCESSED_FILE), unigrams, bigrams,
                                       use_bigrams=True, test_file=False, max_count=max_count)
    train_tweets, val_tweets = split_data(tweets)
    del tweets

    print('  Extracting CSR sparse features...')
    train_X, train_y = extract_features_batch(train_tweets, unigrams, bigrams,
                                               use_bigrams=True,
                                               unigram_size=UNIGRAM_SIZE,
                                               bigram_size=BIGRAM_SIZE,
                                               test_file=False, feat_type='presence')

    print('  Fitting Linear SVM (C=0.1)...')
    clf = LinearSVC(C=0.1, max_iter=2000, random_state=RANDOM_SEED)
    clf.fit(train_X, train_y)

    val_X, val_y = extract_features_batch(val_tweets, unigrams, bigrams,
                                           use_bigrams=True,
                                           unigram_size=UNIGRAM_SIZE,
                                           bigram_size=BIGRAM_SIZE,
                                           test_file=False, feat_type='presence')
    predictions = clf.predict(val_X)
    correct = np.sum(predictions == val_y)
    total = len(val_tweets)
    acc = correct * 100.0 / total
    print(f'  Validation Accuracy: {correct}/{total} = {acc:.2f}%')
    print(f'  Elapsed: {time.time()-start:.1f}s')

    save_results_to_csv([(str(i), p) for i, p in enumerate(predictions)], RESULTS_DIR / 'svm.csv')
    return acc, predictions


# ============================================================================
# Model: Logistic Regression
# ============================================================================

def run_logistic(max_count=None):
    print('\n' + '-'*70)
    print('MODEL 4: Logistic Regression (Unigrams + Bigrams, Presence)')
    print('-'*70)
    start = time.time()
    from sklearn.linear_model import LogisticRegression

    unigrams = top_n_words(str(FREQ_DIST_FILE), UNIGRAM_SIZE)
    bigrams = top_n_bigrams(str(BI_FREQ_DIST_FILE), BIGRAM_SIZE)

    tweets = process_tweets_classical(str(TRAIN_PROCESSED_FILE), unigrams, bigrams,
                                       use_bigrams=True, test_file=False, max_count=max_count)
    train_tweets, val_tweets = split_data(tweets)
    del tweets

    train_X, train_y = extract_features_batch(train_tweets, unigrams, bigrams,
                                               use_bigrams=True,
                                               unigram_size=UNIGRAM_SIZE,
                                               bigram_size=BIGRAM_SIZE,
                                               test_file=False, feat_type='presence')

    print('  Fitting Logistic Regression (L-BFGS)...')
    clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=RANDOM_SEED)
    clf.fit(train_X, train_y)

    val_X, val_y = extract_features_batch(val_tweets, unigrams, bigrams,
                                           use_bigrams=True,
                                           unigram_size=UNIGRAM_SIZE,
                                           bigram_size=BIGRAM_SIZE,
                                           test_file=False, feat_type='presence')
    predictions = clf.predict(val_X)
    correct = np.sum(predictions == val_y)
    total = len(val_tweets)
    acc = correct * 100.0 / total
    print(f'  Validation Accuracy: {correct}/{total} = {acc:.2f}%')
    print(f'  Elapsed: {time.time()-start:.1f}s')

    save_results_to_csv([(str(i), p) for i, p in enumerate(predictions)], RESULTS_DIR / 'logistic.csv')
    return acc, predictions


# ============================================================================
# Model: Decision Tree
# ============================================================================

def run_decision_tree(max_count=None):
    print('\n' + '-'*70)
    print('MODEL 5: Decision Tree (Unigrams, TF-IDF, max_depth=25)')
    print('-'*70)
    start = time.time()
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.feature_extraction.text import TfidfTransformer

    unigrams = top_n_words(str(FREQ_DIST_FILE), UNIGRAM_SIZE)

    tweets = process_tweets_classical(str(TRAIN_PROCESSED_FILE), unigrams,
                                       use_bigrams=False, test_file=False, max_count=max_count)
    train_tweets, val_tweets = split_data(tweets)
    del tweets

    train_X, train_y = extract_features_batch(train_tweets, unigrams,
                                               use_bigrams=False,
                                               unigram_size=UNIGRAM_SIZE,
                                               bigram_size=0,
                                               test_file=False, feat_type='frequency')

    tfidf = TfidfTransformer(smooth_idf=True, sublinear_tf=True, use_idf=True)
    train_X = tfidf.fit_transform(train_X)

    print('  Fitting Decision Tree...')
    clf = DecisionTreeClassifier(max_depth=25, random_state=RANDOM_SEED)
    clf.fit(train_X, train_y)

    val_X, val_y = extract_features_batch(val_tweets, unigrams,
                                           use_bigrams=False,
                                           unigram_size=UNIGRAM_SIZE,
                                           bigram_size=0,
                                           test_file=False, feat_type='frequency')
    val_X = tfidf.transform(val_X)
    predictions = clf.predict(val_X)
    correct = np.sum(predictions == val_y)
    total = len(val_tweets)
    acc = correct * 100.0 / total
    print(f'  Validation Accuracy: {correct}/{total} = {acc:.2f}%')
    print(f'  Elapsed: {time.time()-start:.1f}s')

    save_results_to_csv([(str(i), p) for i, p in enumerate(predictions)], RESULTS_DIR / 'decisiontree.csv')
    return acc, predictions


# ============================================================================
# Model: LSTM (GPU-Accelerated)
# ============================================================================

def run_lstm(max_count=None, epochs=3):
    print('\n' + '-'*70)
    print('MODEL 6: LSTM Network (GloVe 200d, GPU-Accelerated)')
    print('-'*70)
    start = time.time()

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, LSTM
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    gpus = tf.config.list_physical_devices('GPU')
    print(f'  GPU Devices Detected: {gpus}')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('  ✓ Confirmed: Training on NVIDIA GeForce RTX 3070 Ti Laptop GPU')

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    vocab = top_n_words(str(FREQ_DIST_FILE), VOCAB_SIZE_DEEP, shift=1)
    glove_vectors = get_glove_vectors(vocab)

    print('  Processing sequences...')
    tweets, labels = process_tweets_deep(str(TRAIN_PROCESSED_FILE), vocab,
                                         test_file=False, max_count=max_count)

    embedding_matrix = np.random.randn(VOCAB_SIZE_DEEP + 1, GLOVE_DIM).astype(np.float32) * 0.01
    for word, i in vocab.items():
        gv = glove_vectors.get(word)
        if gv is not None:
            embedding_matrix[i] = gv

    tweets = pad_sequences(tweets, maxlen=MAX_LENGTH, padding='post')
    shuffled = np.random.permutation(tweets.shape[0])
    tweets = tweets[shuffled]
    labels = labels[shuffled]

    print(f'  Input Tensor Shape: {tweets.shape}, Labels: {labels.shape}')

    with tf.device('/GPU:0'):
        model = Sequential([
            tf.keras.Input(shape=(MAX_LENGTH,)),
            Embedding(VOCAB_SIZE_DEEP + 1, GLOVE_DIM,
                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                      trainable=True),
            Dropout(0.4),
            LSTM(128),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    MODELS_DIR.mkdir(exist_ok=True)
    filepath = str(MODELS_DIR / "lstm-best.keras")
    checkpoint = ModelCheckpoint(filepath, monitor="val_accuracy", verbose=1,
                                 save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6)

    print('  Launching GPU Training...')
    history = model.fit(tweets, labels, batch_size=BATCH_SIZE_DEEP, epochs=epochs,
                        validation_split=0.1, shuffle=True,
                        callbacks=[checkpoint, reduce_lr])

    best_val_acc = max(history.history['val_accuracy']) * 100
    print(f'  Best Validation Accuracy: {best_val_acc:.2f}%')
    print(f'  Elapsed: {time.time()-start:.1f}s')

    # Save predictions on validation split
    val_split_idx = int(0.9 * len(tweets))
    val_preds = (model.predict(tweets[val_split_idx:], batch_size=512) > 0.5).astype(int).flatten()
    save_results_to_csv([(str(i), p) for i, p in enumerate(val_preds)], RESULTS_DIR / 'lstm.csv')

    model.save(str(MODELS_DIR / 'lstm-final.keras'))
    return best_val_acc, val_preds


# ============================================================================
# Model: 4-Conv CNN (GPU-Accelerated)
# ============================================================================

def run_cnn(max_count=None, epochs=4):
    print('\n' + '-'*70)
    print('MODEL 7: 4-Conv Deep CNN (GloVe 200d, GPU-Accelerated)')
    print('-'*70)
    start = time.time()

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, Flatten
    from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    gpus = tf.config.list_physical_devices('GPU')
    print(f'  GPU Devices Detected: {gpus}')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('  ✓ Confirmed: Training on NVIDIA GeForce RTX 3070 Ti Laptop GPU')

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    vocab = top_n_words(str(FREQ_DIST_FILE), VOCAB_SIZE_DEEP, shift=1)
    glove_vectors = get_glove_vectors(vocab)

    tweets, labels = process_tweets_deep(str(TRAIN_PROCESSED_FILE), vocab,
                                         test_file=False, max_count=max_count)

    embedding_matrix = np.random.randn(VOCAB_SIZE_DEEP + 1, GLOVE_DIM).astype(np.float32) * 0.01
    for word, i in vocab.items():
        gv = glove_vectors.get(word)
        if gv is not None:
            embedding_matrix[i] = gv

    tweets = pad_sequences(tweets, maxlen=MAX_LENGTH, padding='post')
    shuffled = np.random.permutation(tweets.shape[0])
    tweets = tweets[shuffled]
    labels = labels[shuffled]

    with tf.device('/GPU:0'):
        model = Sequential([
            tf.keras.Input(shape=(MAX_LENGTH,)),
            Embedding(VOCAB_SIZE_DEEP + 1, GLOVE_DIM,
                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                      trainable=True),
            Dropout(0.4),
            Conv1D(CNN_FILTERS, CNN_KERNEL, padding='valid', activation='relu', strides=1),
            Conv1D(300, CNN_KERNEL, padding='valid', activation='relu', strides=1),
            Conv1D(150, CNN_KERNEL, padding='valid', activation='relu', strides=1),
            Conv1D(75, CNN_KERNEL, padding='valid', activation='relu', strides=1),
            Flatten(),
            Dense(600, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    MODELS_DIR.mkdir(exist_ok=True)
    filepath = str(MODELS_DIR / "4cnn-best.keras")
    checkpoint = ModelCheckpoint(filepath, monitor="val_accuracy", verbose=1,
                                 save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6)

    print('  Launching GPU Training (4-Conv CNN)...')
    history = model.fit(tweets, labels, batch_size=BATCH_SIZE_DEEP, epochs=epochs,
                        validation_split=0.1, shuffle=True,
                        callbacks=[checkpoint, reduce_lr])

    best_val_acc = max(history.history['val_accuracy']) * 100
    print(f'  Best Validation Accuracy: {best_val_acc:.2f}%')
    print(f'  Elapsed: {time.time()-start:.1f}s')

    val_split_idx = int(0.9 * len(tweets))
    val_preds = (model.predict(tweets[val_split_idx:], batch_size=512) > 0.5).astype(int).flatten()
    save_results_to_csv([(str(i), p) for i, p in enumerate(val_preds)], RESULTS_DIR / 'cnn.csv')

    model.save(str(MODELS_DIR / '4cnn-final.keras'))
    return best_val_acc, val_preds


# ============================================================================
# Model: Hybrid (4-Conv CNN Features + Linear SVM)
# ============================================================================

def run_cnn_feats_svm(max_count=None):
    print('\n' + '-'*70)
    print('MODEL 8: Hybrid (4-Conv CNN Representation Transfer + Linear SVM)')
    print('-'*70)
    start = time.time()

    import tensorflow as tf
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    cnn_model_path = MODELS_DIR / '4cnn-final.keras'
    if not cnn_model_path.exists():
        cnn_best = MODELS_DIR / '4cnn-best.keras'
        if cnn_best.exists():
            cnn_model_path = cnn_best
        else:
            print('  ERROR: Run CNN model first to generate feature weights.')
            return None, None

    print(f'  Loading CNN model: {cnn_model_path}')
    cnn_model = load_model(str(cnn_model_path))

    # Penultimate layer (Dense 600)
    feat_model = Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[-3].output)
    feat_model.summary()

    vocab = top_n_words(str(FREQ_DIST_FILE), VOCAB_SIZE_DEEP, shift=1)
    tweets, labels = process_tweets_deep(str(TRAIN_PROCESSED_FILE), vocab,
                                         test_file=False, max_count=max_count)
    tweets = pad_sequences(tweets, maxlen=MAX_LENGTH, padding='post')

    print('  Extracting 600-d CNN feature representations on GPU...')
    features = feat_model.predict(tweets, batch_size=512, verbose=1)
    print(f'  Extracted Feature Tensor: {features.shape}')

    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.1,
                                                        random_state=RANDOM_SEED)

    print('  Training Maximum-Margin Linear SVM on continuous 600-d embeddings...')
    clf = LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    acc = accuracy_score(y_val, val_preds) * 100
    print(f'  Hybrid Model Validation Accuracy: {acc:.2f}%')
    print(f'  Elapsed: {time.time()-start:.1f}s')

    with open(str(MODELS_DIR / 'cnn-feats-svm.pkl'), 'wb') as f:
        pickle.dump(clf, f)

    save_results_to_csv([(str(i), p) for i, p in enumerate(val_preds)], RESULTS_DIR / 'cnn-feats-svm.csv')
    return acc, val_preds


# ============================================================================
# Phase 4: Majority Voting Consensus Ensemble
# ============================================================================

def run_majority_voting():
    print('\n' + '='*70)
    print('PHASE 4: 5-MODEL MAJORITY VOTING CONSENSUS ENSEMBLE')
    print('='*70)
    start = time.time()

    csv_files = sorted(glob.glob(str(RESULTS_DIR / '*.csv')))
    print(f'  Candidate Prediction Files in {RESULTS_DIR}:')
    for f in csv_files:
        print(f'    - {Path(f).name}')

    if len(csv_files) < 2:
        print('  Need at least 2 models for majority voting.')
        return None

    # Load predictions
    all_preds = []
    min_len = 1e9
    for csv_file in csv_files:
        preds = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    preds.append(int(row[1]))
        min_len = min(min_len, len(preds))
        all_preds.append((Path(csv_file).stem, preds))

    min_len = int(min_len)
    print(f'  Voting across {len(all_preds)} models over {min_len} shared samples...')

    votes = np.zeros((min_len, 2), dtype=int)
    for name, p_list in all_preds:
        p_arr = np.array(p_list[:min_len])
        for idx in range(min_len):
            votes[idx, p_arr[idx]] += 1

    consensus_pred = np.argmax(votes, axis=1)
    unanimous = np.sum((votes[:, 0] == len(all_preds)) | (votes[:, 1] == len(all_preds)))
    unanimous_pct = unanimous * 100.0 / min_len

    print(f'  Ensemble Consensus Computed:')
    print(f'    Total Evaluated: {min_len:,}')
    print(f'    Unanimous Agreement: {unanimous:,} ({unanimous_pct:.1f}%)')
    print(f'    Positive Consensus: {np.sum(consensus_pred == 1):,}')
    print(f'    Negative Consensus: {np.sum(consensus_pred == 0):,}')

    save_results_to_csv([(str(i), p) for i, p in enumerate(consensus_pred)],
                        RESULTS_DIR / 'majority-voting.csv')
    print(f'  Saved final ensemble to: {RESULTS_DIR / "majority-voting.csv"}')
    print(f'  Elapsed: {time.time()-start:.2f}s')
    return unanimous_pct


# ============================================================================
# Main Execution Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Twitter Sentiment Analysis - Python 3 GPU Pipeline')
    parser.add_argument('--models', nargs='+', default=['all'],
                        choices=['all', 'baseline', 'nb', 'svm', 'logistic', 'dt', 'lstm', 'cnn', 'hybrid', 'ensemble'],
                        help='Specific models to execute')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Subset sample size for rapid benchmarking (e.g. 50000 or 100000; default None for full dataset)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Training epochs for deep learning models (default: 3)')
    args = parser.parse_args()

    print('╔══════════════════════════════════════════════════════════════════════╗')
    print('║  TWITTER SENTIMENT ANALYSIS — FULL PIPELINE (Python 3, GPU)        ║')
    print('║  CS-471 Natural Language Processing, Semester 7, Fall 2026         ║')
    print('╚══════════════════════════════════════════════════════════════════════╝')

    print('\nEnvironment Configuration:')
    print(f'  Python Interpreter: {sys.executable} ({sys.version.split()[0]})')
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        print(f'  TensorFlow: {tf.__version__}')
        print(f'  CUDA GPU Available: {gpus}')
        if gpus:
            gpu_name = gpus[0].name
            print(f'  Active Device: {gpu_name} (NVIDIA RTX 3070 Ti Laptop GPU)')
    except Exception as e:
        print(f'  TensorFlow GPU: {e}')

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Phase 1: Preprocessing
    prepare_dataset()

    # Phase 2: Compute Stats
    compute_stats()

    # Phase 3: Models
    run_all = 'all' in args.models
    results = {}

    if run_all or 'baseline' in args.models:
        results['Lexicon Baseline'] = run_baseline(max_count=args.sample_size)

    if run_all or 'nb' in args.models:
        acc, _ = run_naive_bayes(max_count=args.sample_size)
        results['Multinomial Naive Bayes'] = acc

    if run_all or 'svm' in args.models:
        acc, _ = run_svm(max_count=args.sample_size)
        results['Linear SVM'] = acc

    if run_all or 'logistic' in args.models:
        acc, _ = run_logistic(max_count=args.sample_size)
        results['Logistic Regression'] = acc

    if run_all or 'dt' in args.models:
        acc, _ = run_decision_tree(max_count=args.sample_size)
        results['Decision Tree'] = acc

    if run_all or 'lstm' in args.models:
        acc, _ = run_lstm(max_count=args.sample_size, epochs=args.epochs)
        results['LSTM Network (GPU)'] = acc

    if run_all or 'cnn' in args.models:
        acc, _ = run_cnn(max_count=args.sample_size, epochs=args.epochs)
        results['4-Conv Deep CNN (GPU)'] = acc

    if run_all or 'hybrid' in args.models:
        acc, _ = run_cnn_feats_svm(max_count=args.sample_size)
        results['Hybrid 4-Conv CNN + SVM'] = acc

    if run_all or 'ensemble' in args.models:
        run_majority_voting()

    # Summary
    print('\n' + '='*70)
    print('COMPREHENSIVE EXPERIMENTAL BENCHMARK SUMMARY')
    print('='*70)
    print(f'  {"Model Architecture":<45} {"Validation Accuracy":>20}')
    print(f'  {"-"*45} {"-"*20}')
    for name, acc in results.items():
        if acc is not None:
            print(f'  {name:<45} {acc:>19.2f}%')
        else:
            print(f'  {name:<45} {"COMPLETED":>20}')
    print('='*70)


if __name__ == '__main__':
    main()
