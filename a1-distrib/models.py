# models.py

from sentiment_data import *
from utils import *
from collections import Counter
import numpy as np
import nltk
from nltk.corpus import stopwords
np.random.seed(2025)

# Helper function to compute accuracy
def compute_accuracy(model, examples):
    correct = 0
    for ex in examples:
        pred = model.predict(ex.words)
        if pred == ex.label:
            correct += 1
    return correct / len(examples) if examples else 0.0

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
        raise Exception("Must be implemented")


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


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
        self.indexer = feat_extractor.get_indexer()
        self.feat_extractor = feat_extractor
        self.weights = Counter()

    def predict(self, sentence: List[str]) -> int:
        feature_vector = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = 0
        for feature, count in feature_vector.items():
            if feature in self.weights:
                score += self.weights[feature] * count

        return int(score >= 0)


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self):
        raise Exception("Must be implemented")


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
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
            for feature, feature_count in feature_vector.items():
                if feature in perceptron_model.weights:
                    score += perceptron_model.weights[feature] * feature_count
            predicted_label = int(score >= 0)
            for feature, feature_count in feature_vector.items():
                weight_update = (example.label - predicted_label) * feature_count
                perceptron_model.weights[feature] += weight_update
        # Compute and print training accuracy after each epoch
        # train_acc = compute_accuracy(perceptron_model, train_exs)
        # print(f"Epoch {epoch+1}: Training accuracy = {train_acc:.4f}")
    return perceptron_model



def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    raise Exception("Must be implemented")


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
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model