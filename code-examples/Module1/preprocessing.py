# Text preprocessing pipeline for paragraphs
# Features:
# - tokenization
# - stopword removal (toggle)
# - casing (lower, preserve) (toggle)
# - unknown word handling
# - indexing to integer IDs

import re
from collections import defaultdict

# --- 1) configuration ---
REMOVE_STOPWORDS = True  # set False to keep stopwords
LOWERCASE = True          # set False to preserve case
UNK_TOKEN = "<UNK>"

# simple stopword list (expand as needed)
STOPWORDS = {"the","was","is","and","i","to","it","a","of","in","on"}

# --- 2) preprocessing function ---
def preprocess(texts, vocab=None):
    """
    texts: list of paragraphs (strings)
    vocab: optional existing vocab dict {token: index}
    returns: tokenized texts, vocab, indexed sequences
    """
    tokenized = []
    for para in texts:
        # split on non-alphabetic chars
        tokens = re.findall(r"\b\w+\b", para)
        if LOWERCASE:
            tokens = [t.lower() for t in tokens]
        if REMOVE_STOPWORDS:
            tokens = [t for t in tokens if t not in STOPWORDS]
        tokenized.append(tokens)

    # build vocab if not provided
    if vocab is None:
        vocab = {UNK_TOKEN:0}
        for toks in tokenized:
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)

    # map tokens to IDs (use <UNK> if unseen)
    indexed = []
    for toks in tokenized:
        ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in toks]
        indexed.append(ids)

    return tokenized, vocab, indexed

# --- 3) demo with paragraphs ---
paragraphs = [
    "Yash Gajera went to the park. The movie was good and fun!",
    "In New York, I tried pizza. It was absolutely fantastic!",
    "He appears here with friends and family often.",
    "With strong skills, she appears confident in her work.",
    "Here is another line with more words than before."
]


tokens, vocab, ids = preprocess(paragraphs)

# Print configuration
print(f"REMOVE_STOPWORDS={REMOVE_STOPWORDS}, LOWERCASE={LOWERCASE}\n")

print("Tokenized:")
for t in tokens:
    print(t)

print("\nVocab mapping (token -> id):")
print(vocab)

print("\nIndexed sequences:")
for i in ids:
    print(i)

# --- 4) show handling of unknown word ---
new_texts = ["RandomWordNotInVocab appears here with Gajera."]
_, _, new_ids = preprocess(new_texts, vocab)
print("\nIndexed with UNK handling:")
print(new_ids)
