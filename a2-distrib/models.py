# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from nltk import edit_distance

# Set random seeds for repeatability
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)

# Global Hyperparameters
DEFAULT_HIDDEN_SIZE = 100 # 100
DEFAULT_EPOCHS = 10 # 10
DEFAULT_LR = 1e-3 # 0.001
DEFAULT_BATCH_SIZE = 128 # 1

class DAN(nn.Module):
    def __init__(self, embedding_size, hidden_layer_size, output_size):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param embedding_size: size of input (integer)
        :param hidden_layer_size: size of hidden layer (integer)
        :param output_size: size of output (integer), which should be the number of classes
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, hidden_layer_size, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, output_size, dtype=torch.float64),
        )

    def forward(self, input_tensor):
        return self.net(input_tensor)
class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, dan_model, word_embeddings, device: str | None = None, word_dictionary: dict | None = None):
        self.model = dan_model
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(device)
        self.word_embeddings = word_embeddings
        self.device = device
        self.word_dictionary = word_dictionary or {}

    def _maybe_correct_word(self, word: str) -> str:
        """
        Returns a corrected word if possible using the simple prefix+length buckets and edit distance 1;
        otherwise returns the original word.
        """
        if not self.word_dictionary or self.word_embeddings.word_indexer.index_of(word) != -1:
            return word
        prefix = word[:3]
        length = len(word)
        if prefix in self.word_dictionary and length in self.word_dictionary[prefix]:
            for candidate_word in self.word_dictionary[prefix][length]:
                if edit_distance(word, candidate_word) == 1:
                    return candidate_word
        return word

    def _average_embeddings(self, words: List[str], has_typos: bool) -> np.ndarray:
        """
        Returns the mean embedding for a list of words, correcting typos if needed.
        Returns a zero vector if the list is empty.
        """
        if len(words) == 0:
            return np.zeros(self.word_embeddings.get_embedding_length())
        if has_typos and self.word_dictionary:
            corrected = [self._maybe_correct_word(w) for w in words]
            vectors = [self.word_embeddings.get_embedding(w) for w in corrected]
        else:
            vectors = [self.word_embeddings.get_embedding(w) for w in words]
        return np.mean(np.array(vectors), axis=0)

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Predicts the sentiment label for a list of words, optionally correcting typos.
        Returns the predicted class as an integer.
        """
        average_vector = self._average_embeddings(ex_words, has_typos)

        model_dtype = next(self.model.parameters()).dtype
        input_tensor = torch.tensor(average_vector, dtype=model_dtype).to(self.device)
        with torch.no_grad():
            logits = self.model(input_tensor)
            return logits.argmax(dim=0).item()

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        Predicts sentiment labels for a batch of sentences using the trained model.
        Processes sentences in batches for faster inference.
        """
        predictions: List[int] = []
        embedding_size = self.word_embeddings.get_embedding_length()
        model_dtype = next(self.model.parameters()).dtype
        for batch_start in range(0, len(all_ex_words), DEFAULT_BATCH_SIZE):
            batch_sentences = all_ex_words[batch_start:batch_start+DEFAULT_BATCH_SIZE]
            batch_vectors = [self._average_embeddings(sentence_words, has_typos) for sentence_words in batch_sentences]
            input_batch = torch.tensor(np.stack(batch_vectors, axis=0), dtype=model_dtype).to(self.device)
            with torch.no_grad():
                logits = self.model(input_batch)  # (B, C)
                batch_predictions = logits.argmax(dim=1).tolist()
                predictions.extend(batch_predictions)
        return predictions


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    embedding_size = word_embeddings.get_embedding_length()
    output_size = 2  # binary sentiment
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DAN(embedding_size, DEFAULT_HIDDEN_SIZE, output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=DEFAULT_LR)
    criterion = nn.CrossEntropyLoss()

    model_dtype = next(model.parameters()).dtype

    word_dictionary: dict | None = None
    if train_model_for_typo_setting:
        tmp_dictionary: dict = {}
        for example in train_exs:
            for word in example.words:
                if len(word) < 4:
                    continue
                prefix = word[:3]
                length = len(word)
                if prefix not in tmp_dictionary:
                    tmp_dictionary[prefix] = {}
                if length not in tmp_dictionary[prefix]:
                    tmp_dictionary[prefix][length] = set()
                tmp_dictionary[prefix][length].add(word)
        word_dictionary = {p: {l: list(words) for l, words in lens.items()} for p, lens in tmp_dictionary.items()}

    for _ in range(DEFAULT_EPOCHS):
        random.shuffle(train_exs)
        for batch_start in range(0, len(train_exs), DEFAULT_BATCH_SIZE):
            batch_examples = train_exs[batch_start:batch_start+DEFAULT_BATCH_SIZE]
            batch_vectors = []
            batch_labels = []
            for example in batch_examples:
                if len(example.words) == 0:
                    average_vector = np.zeros(embedding_size)
                else:
                    word_vectors = np.array([word_embeddings.get_embedding(word) for word in example.words])
                    average_vector = np.mean(word_vectors, axis=0)
                batch_vectors.append(average_vector)
                batch_labels.append(example.label)

            input_batch = torch.tensor(np.stack(batch_vectors, axis=0), dtype=model_dtype).to(device)
            targets = torch.tensor(batch_labels, dtype=torch.long).to(device)

            model.train()
            optimizer.zero_grad()
            logits = model(input_batch)
            loss_value = criterion(logits, targets)
            loss_value.backward()
            optimizer.step()

    return NeuralSentimentClassifier(model, word_embeddings, device, word_dictionary)

