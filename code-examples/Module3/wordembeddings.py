# pip install gensim==4.3.2
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.decomposition import PCA
import numpy as np
import random

# --- toy corpus ---
raw = [
    "the movie was good and fun",
    "the film was great and enjoyable",
    "i watched the movie yesterday",
    "i watched the film today",
    "the movie inspired me",
    "the film inspired me",
    "the mango was ripe",
    "cash money business news today"
]
corpus = [simple_preprocess(s) for s in raw]

# --- reproducibility ---
SEED = 42
np.random.seed(SEED); random.seed(SEED)

# --- train Word2Vec ---
# sg=0 => CBOW, sg=1 => Skip-gram
model = Word2Vec(
    sentences=corpus,
    vector_size=50,
    window=3,
    min_count=1,
    workers=1,
    sg=1,            # set 0 for CBOW, 1 for Skip-gram
    negative=5,
    epochs=50,
    seed=SEED
)

# --- inspect vocabulary ---
print("Vocab:", sorted(model.wv.index_to_key))

# --- nearest neighbors ---
for w in ["movie","film","good","mango","cash"]:
    print(f"\nNearest to '{w}':")
    for nbr, score in model.wv.most_similar(w, topn=5):
        print(f"  {nbr:>10s}  sim={score:.3f}")

# --- simple analogy: movie ≈ film ---
# vector("movie") - vector("film") + vector("great") ~ ?
print("\nAnalogy: movie - film + great =>")
for w, s in model.wv.most_similar(positive=["movie","great"], negative=["film"], topn=5):
    print(f"  {w:>10s}  score={s:.3f}")

# --- sentence similarity via averaged embeddings ---
def sent_vec(tokens):
    vs = [model.wv[w] for w in tokens if w in model.wv]
    return np.mean(vs, axis=0) if vs else np.zeros(model.vector_size)

s1 = sent_vec("movie was good".split())
s2 = sent_vec("film was great".split())
cos = float(s1 @ s2 / (np.linalg.norm(s1)*np.linalg.norm(s2) + 1e-12))
print("\nCosine(movie was good, film was great) =", round(cos, 3))

# --- save / load ---
model.save("toy_word2vec.model")
loaded = Word2Vec.load("toy_word2vec.model")
print("\nLoaded OK. Dim =", loaded.wv.vector_size)
