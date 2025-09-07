import numpy as np
from tokenizer import tokenize
from collections import Counter
import re
from spacy.lang.en import English
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Test the tokenizer
    #print(repr(tokenize("said.")))
    file = open("a0/nyt.txt", "r")
    sentences = file.readlines()
    file.close()

    white_space_tokenizer = lambda x: x.split()
    white_space_sentences = [white_space_tokenizer(sentence) for sentence in sentences]
    wsCounter = Counter()
    for sentence in white_space_sentences:
        wsCounter.update(sentence)
    print("10 most common tokens (whitespace tokenizer):", wsCounter.most_common(10))

    tokenized_sentences = [tokenize(sentence) for sentence in sentences]
    tokenized_counter = Counter()
    for sentence in tokenized_sentences:
        tokenized_counter.update(sentence)

    print("10 most common tokens:", tokenized_counter.most_common(10))
    
    nlp = English()
    spacy_tokenizer = nlp.tokenizer
    tok_sent = [spacy_tokenizer(sentence) for sentence in sentences]
    spacy_counter = Counter()
    for sentence in tok_sent:
        spacy_counter.update([str(token) for token in sentence])
    print("10 most common tokens (spacy):", spacy_counter.most_common(10))

    # Plotting the token frequency distributions
    ws_freqs = [freq for token, freq in wsCounter.most_common()]
    tokenized_freqs = [freq for token, freq in tokenized_counter.most_common()]
    spacy_freqs = [freq for token, freq in spacy_counter.most_common()]
    plt.figure(figsize=(10, 6))
    plt.loglog(ws_freqs, label='Whitespace Tokenizer')
    plt.loglog(tokenized_freqs, label='Custom Tokenizer')
    plt.loglog(spacy_freqs, label='Spacy Tokenizer')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Token Frequency Distributions')
    plt.legend()
    plt.grid(True)
    plt.show()
