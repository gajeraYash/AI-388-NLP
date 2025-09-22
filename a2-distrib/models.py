# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *

# Set random seeds for repeatability
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)

# =====================
# Global Hyperparameters
# =====================
DEFAULT_HIDDEN_SIZE = 256 # 100
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
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)
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
    def __init__(self, dan_model, word_embeddings, device: str | None = None):
        self.model = dan_model
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(device)
        self.word_embeddings = word_embeddings
        self.device = device

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        # Average the embeddings for all words in the example
        if len(ex_words) == 0:
            avg_vec = np.zeros(self.word_embeddings.get_embedding_length())
        else:
            word_vecs = np.array([self.word_embeddings.get_embedding(w) for w in ex_words])
            avg_vec = np.mean(word_vecs, axis=0)

        model_dtype = next(self.model.parameters()).dtype
        x = torch.tensor(avg_vec, dtype=model_dtype).to(self.device)
        with torch.no_grad():
            out = self.model(x)
            return out.argmax(dim=0).item()

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        preds: List[int] = []
        embedding_size = self.word_embeddings.get_embedding_length()
        model_dtype = next(self.model.parameters()).dtype
        for i in range(0, len(all_ex_words), DEFAULT_BATCH_SIZE):
            batch_sents = all_ex_words[i:i+DEFAULT_BATCH_SIZE]
            batch_vecs = []
            for words in batch_sents:
                if len(words) == 0:
                    avg_vec = np.zeros(embedding_size)
                else:
                    word_vecs = np.array([self.word_embeddings.get_embedding(w) for w in words])
                    avg_vec = np.mean(word_vecs, axis=0)
                batch_vecs.append(avg_vec)
            X = torch.tensor(np.stack(batch_vecs, axis=0), dtype=model_dtype).to(self.device)
            with torch.no_grad():
                out = self.model(X)  # (B, C)
                batch_preds = out.argmax(dim=1).tolist()
                preds.extend(batch_preds)
        return preds


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
    # Simple hyperparameters
    embedding_size = word_embeddings.get_embedding_length()
    output_size = 2  # binary sentiment
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model, optimizer, loss
    model = DAN(embedding_size, DEFAULT_HIDDEN_SIZE, output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=DEFAULT_LR)
    criterion = nn.CrossEntropyLoss()

    model_dtype = next(model.parameters()).dtype

    # Train with mini-batches over averaged embeddings
    for _ in range(DEFAULT_EPOCHS):
        random.shuffle(train_exs)
        for i in range(0, len(train_exs), DEFAULT_BATCH_SIZE):
            batch = train_exs[i:i+DEFAULT_BATCH_SIZE]
            batch_vecs = []
            batch_labels = []
            for ex in batch:
                if len(ex.words) == 0:
                    avg_vec = np.zeros(embedding_size)
                else:
                    word_vecs = np.array([word_embeddings.get_embedding(w) for w in ex.words])
                    avg_vec = np.mean(word_vecs, axis=0)
                batch_vecs.append(avg_vec)
                batch_labels.append(ex.label)

            X = torch.tensor(np.stack(batch_vecs, axis=0), dtype=model_dtype).to(device)
            y = torch.tensor(batch_labels, dtype=torch.long).to(device)

            model.train()
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    # Wrap in classifier for inference
    return NeuralSentimentClassifier(model, word_embeddings, device)

