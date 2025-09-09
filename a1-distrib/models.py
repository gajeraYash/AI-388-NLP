# models.py

from sentiment_data import *
from utils import *
from collections import Counter
import numpy as np
import nltk
from nltk.corpus import stopwords
np.random.seed(2025)

# Helper functions
def compute_accuracy(model, examples):
    correct = 0
    for ex in examples:
        pred = model.predict(ex.words)
        if pred == ex.label:
            correct += 1
    return correct / len(examples) if examples else 0.0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = set(stopwords.words('english'))
        self.size = len(indexer)

    def get_indexer(self):
        return self.indexer
    
    def get_size(self):
        return self.size
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        feature_vector = Counter()
        for word in sentence:
            word = word.lower()
            if word not in self.stop_words:
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(word)
                else:
                    idx = self.indexer.index_of(word)
                feature_vector[idx] += 1
        
        return feature_vector



class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.size = len(indexer)

    def get_indexer(self):
        return self.indexer

    def get_size(self):
        return self.size

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        feature_vector = Counter()
        # Extract bigram features: indicators for adjacent pairs
        for word1, word2 in zip(sentence, sentence[1:]):
            bigram = (word1.lower(), word2.lower())
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(bigram)
            else:
                idx = self.indexer.index_of(bigram)
            feature_vector[idx] += 1
        return feature_vector


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stop_words = set(stopwords.words('english'))
        self.size = len(indexer)
        self.rare_words = set()
        self.idf = {}

    def get_indexer(self):
        return self.indexer

    def get_size(self):
        return self.size

    def _compute_idf(self, num_docs, doc_freq):
        return np.log((1 + num_docs) / (1 + doc_freq)) + 1

    def build_stats(self, train_exs, min_freq=2):
        # Build set of rare words and compute IDF for tf-idf
        word_counter = Counter()
        bigram_counter = Counter()
        trigram_counter = Counter()
        word_doc_counter = Counter()
        bigram_doc_counter = Counter()
        trigram_doc_counter = Counter()
        num_docs = len(train_exs)
        for example in train_exs:
            words_in_doc = set()
            bigrams_in_doc = set()
            trigrams_in_doc = set()
            sentence = example.words
            for word in sentence:
                word_lc = word.lower()
                if word_lc not in self.stop_words:
                    word_counter[word_lc] += 1
                    words_in_doc.add(word_lc)
            for i in range(len(sentence) - 1):
                w1 = sentence[i].lower()
                w2 = sentence[i+1].lower()
                if w1 not in self.stop_words and w2 not in self.stop_words:
                    bigram = (w1, w2)
                    bigram_counter[bigram] += 1
                    bigrams_in_doc.add(bigram)
            for i in range(len(sentence) - 2):
                w1 = sentence[i].lower()
                w2 = sentence[i+1].lower()
                w3 = sentence[i+2].lower()
                if w1 not in self.stop_words and w2 not in self.stop_words and w3 not in self.stop_words:
                    trigram = (w1, w2, w3)
                    trigram_counter[trigram] += 1
                    trigrams_in_doc.add(trigram)
            for w in words_in_doc:
                word_doc_counter[w] += 1
            for b in bigrams_in_doc:
                bigram_doc_counter[b] += 1
            for t in trigrams_in_doc:
                trigram_doc_counter[t] += 1
        self.rare_words = set([w for w, c in word_counter.items() if c < min_freq])
        # Compute IDF for words, bigrams, trigrams
        self.idf = {}
        for w in word_doc_counter:
                self.idf[('uni', w)] = self._compute_idf(num_docs, word_doc_counter[w])
        for b in bigram_doc_counter:
                self.idf[('bi', b)] = self._compute_idf(num_docs, bigram_doc_counter[b])
        for t in trigram_doc_counter:
                self.idf[('tri', t)] = self._compute_idf(num_docs, trigram_doc_counter[t])

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        feature_vector = Counter()
        term_freq = Counter()
        # Unigram term frequency
        for word in sentence:
            word_lc = word.lower()
            if word_lc in self.stop_words or word_lc in self.rare_words:
                continue
            term_freq[('uni', word_lc)] += 1
        # Bigram term frequency
        for i in range(len(sentence) - 1):
            w1 = sentence[i].lower()
            w2 = sentence[i+1].lower()
            if w1 in self.stop_words or w2 in self.stop_words or w1 in self.rare_words or w2 in self.rare_words:
                continue
            bigram = (w1, w2)
            term_freq[('bi', bigram)] += 1
        # Trigram term frequency
        for i in range(len(sentence) - 2):
            w1 = sentence[i].lower()
            w2 = sentence[i+1].lower()
            w3 = sentence[i+2].lower()
            if w1 in self.stop_words or w2 in self.stop_words or w3 in self.stop_words or w1 in self.rare_words or w2 in self.rare_words or w3 in self.rare_words:
                continue
            trigram = (w1, w2, w3)
            term_freq[('tri', trigram)] += 1
        # Apply tf-idf weighting and clip frequency at 3
        for feature, freq in term_freq.items():
            if add_to_indexer:
                feature_index = self.indexer.add_and_get_index(feature)
            else:
                feature_index = self.indexer.index_of(feature)
            idf_val = self.idf.get(feature, 1.0)
            feature_vector[feature_index] = min(freq, 3) * idf_val
        return feature_vector


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feat_extractor: FeatureExtractor):
        self.feat_extractor = feat_extractor
        self.weights = Counter()

    def predict(self, sentence: List[str]) -> int:
        feature_vector = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = 0
        for feature_index, feature_value in feature_vector.items():
            if feature_index in self.weights:
                score += self.weights[feature_index] * feature_value
        return int(score >= 0)


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feat_extractor: FeatureExtractor):
        self.feat_extractor = feat_extractor
        self.weights = np.zeros(len(feat_extractor.get_indexer()))

    def predict(self, sentence: List[str]) -> int:
        feats = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        # Convert sparse Counter to dense numpy array
        x = np.zeros_like(self.weights)
        for idx, value in feats.items():
            x[idx] = value
        score = np.dot(self.weights, x)
        probability = sigmoid(score)
        return int(probability >= 0.5)


def train_perceptron(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: development set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    perceptron_model = PerceptronClassifier(feat_extractor)
    num_epochs = 5
    for epoch in range(num_epochs):
        np.random.shuffle(train_exs)
        for example in train_exs:
            feature_vector = feat_extractor.extract_features(example.words, add_to_indexer=True)
            score = 0
            for feature_index, feature_value in feature_vector.items():
                if feature_index in perceptron_model.weights:
                    score += perceptron_model.weights[feature_index] * feature_value
            predicted_label = int(score >= 0)
            for feature_index, feature_value in feature_vector.items():
                weight_update = (example.label - predicted_label) * feature_value
                perceptron_model.weights[feature_index] += weight_update
        # Compute and print training accuracy after each epoch
        # dev_acc = compute_accuracy(perceptron_model, dev_exs)
        # print(f"Epoch {epoch+1}: Dev Accuracy = {dev_acc:.4f}")
    return perceptron_model



def train_logistic_regression(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # Build feature space once before training
    for example in train_exs:
        feat_extractor.extract_features(example.words, add_to_indexer=True)

    logistic_model = LogisticRegressionClassifier(feat_extractor)
    learning_rate = 0.1
    num_epochs = 14
    for epoch in range(num_epochs):
        np.random.shuffle(train_exs)
        for example in train_exs:
            feature_vector = feat_extractor.extract_features(example.words, add_to_indexer=False)
            feature_array = np.zeros_like(logistic_model.weights)
            for feature_index, feature_value in feature_vector.items():
                feature_array[feature_index] = feature_value
            score = np.dot(logistic_model.weights, feature_array)
            probability = sigmoid(score)
            error = example.label - probability
            logistic_model.weights += learning_rate * error * feature_array
        # Compute dev set accuracy after each epoch
        # dev_acc = compute_accuracy(logistic_model, dev_exs)
        # print(f"Epoch {epoch+1}: Dev Accuracy = {dev_acc:.4f}")
    return logistic_model


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, dev_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, dev_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model