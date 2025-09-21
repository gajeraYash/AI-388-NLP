# NLP embeddings mini-lab: Skip-gram (SGNS), SPPMI+SVD (GloVe connection), FastText, and a bias axis probe.
# References: skip-gram objective and training; SGNS↔PMI/SPPMI; bias axis and debias attempts.

# pip install gensim==4.3.2 scikit-learn==1.4.2
from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess
from collections import Counter, defaultdict
import numpy as np

# -------------------------
# 0) Toy corpus
# -------------------------
raw = [
    "the movie was good and fun",
    "the film was great and enjoyable",
    "i watched the movie yesterday",
    "i watched the film today",
    "the movie inspired me",
    "the film inspired me",
    "the mango was ripe",
    "cash money business news today",
    "she is a doctor and he is a engineer",
    "she is a nurse and he is a manager",
]
corpus = [simple_preprocess(s) for s in raw]
Vocab = sorted({w for sent in corpus for w in sent})
print("Vocab size:", len(Vocab))

# -------------------------
# 1) Skip-gram with Negative Sampling (SGNS)
#    Efficient training objective for skip-gram.
# -------------------------
sgns = Word2Vec(
    sentences=corpus,
    vector_size=50,
    window=3,
    min_count=1,
    sg=1,          # 1 = skip-gram, 0 = CBOW
    negative=5,    # negative sampling
    epochs=80,
    workers=1,
    seed=42,
)
W2V = sgns.wv

def nn(model, w, k=5):
    return model.most_similar(w, topn=k)

print("\n[SGNS] nearest to 'movie':", nn(W2V, "movie"))
print("[SGNS] nearest to 'film':", nn(W2V, "film"))

# -------------------------
# 2) SPPMI + SVD (matrix view; connects SGNS to shifted PMI).
#    Steps:
#      a) build symmetric co-occurrence counts with window
#      b) compute PMI(i,j) = log p(i,j)/(p(i)p(j))
#      c) Shift by log(k) and clamp: SPPMI = max(PMI - log(k), 0)
#      d) take truncated SVD(SPPMI) to get embeddings
# -------------------------
word2id = {w:i for i,w in enumerate(Vocab)}
id2word = {i:w for w,i in word2id.items()}

window = 3
C = Counter()
unigram = Counter()

for sent in corpus:
    idx = [word2id[w] for w in sent]
    for t, i in enumerate(idx):
        unigram[i] += 1
        L = max(0, t-window)
        R = min(len(idx), t+window+1)
        for jpos in range(L, R):
            if jpos == t: 
                continue
            j = idx[jpos]
            C[(i,j)] += 1

V = len(Vocab)
total_pairs = sum(C.values())
total_unigrams = sum(unigram.values())

# probabilities
p_i  = np.array([unigram[i]/total_unigrams for i in range(V)])
logk = np.log(5)  # use same k as SGNS negatives

# build sparse SPPMI
rows, cols, vals = [], [], []
for (i,j), cij in C.items():
    p_ij = cij / total_pairs
    pmi = np.log(p_ij / (p_i[i]*p_i[j] + 1e-12) + 1e-12)
    sppmi = max(pmi - logk, 0.0)
    if sppmi > 0:
        rows.append(i); cols.append(j); vals.append(sppmi)

# dense for tiny toy data; for big data use sparse SVD
M = np.zeros((V,V), dtype=np.float32)
for r,c,v in zip(rows,cols,vals):
    M[r,c] = v

# SVD to K dims
K = 50
U, S, VT = np.linalg.svd(M, full_matrices=False)
E_sppmi = U[:, :K] * np.sqrt(S[:K])  # common scaling

def nn_sppmi(w, k=5):
    if w not in word2id: 
        return []
    x = E_sppmi[word2id[w]]
    sims = []
    for j in range(V):
        if j == word2id[w]: 
            continue
        y = E_sppmi[j]
        s = float(x @ y) / (np.linalg.norm(x)*np.linalg.norm(y) + 1e-12)
        sims.append((id2word[j], s))
    sims.sort(key=lambda t: t[1], reverse=True)
    return sims[:k]

print("\n[SPPMI/SVD] nearest to 'movie':", nn_sppmi("movie"))
print("[SPPMI/SVD] nearest to 'film':", nn_sppmi("film"))

# -------------------------
# 3) FastText-style subword embeddings (robust to OOV).
# -------------------------
ft = FastText(
    sentences=corpus,
    vector_size=50,
    window=3,
    min_count=1,
    sg=1,          # skip-gram objective underneath
    negative=5,
    epochs=80,
    workers=1,
    seed=42,
)
FT = ft.wv

print("\n[FastText] nearest to 'movie':", FT.most_similar("movie"))
# Out-of-vocabulary probe: composed from n-grams
print("[FastText] OOV vector exists? 'moviefilm' in vocab ->", "moviefilm" in FT.key_to_index)
print("[FastText] cosine(movie, 'moviefilm') ~", float(FT.similarity("movie", "moviefilm")))

# -------------------------
# 4) Simple bias axis probe (he–she).
#    Compute axis = e('he') - e('she'), then show projections of occupations.
#    This illustrates encoded associations; debiasing by projection removal has limits.
# -------------------------
def bias_axis(model, a="he", b="she"):
    return model[a] - model[b]

axis = bias_axis(W2V, "he", "she")
def proj_score(model, w, axis):
    v = model[w] / (np.linalg.norm(model[w]) + 1e-12)
    a = axis / (np.linalg.norm(axis) + 1e-12)
    return float(v @ a)

occupations = [w for w in ["doctor","engineer","nurse","manager","homemaker"] if w in W2V]
bias_scores = {w: proj_score(W2V, w, axis) for w in occupations}
print("\n[Bias probe | + = closer to 'he']:", bias_scores)

# Optional "neutralize" by removing projection onto axis (didactic; not a full fix).
def neutralize(model, w, axis):
    v = model[w]
    a = axis / (np.linalg.norm(axis) + 1e-12)
    return v - (v @ a) * a

neutral_scores = {}
for w in occupations:
    v_neu = neutralize(W2V, w, axis)
    neutral_scores[w] = float((v_neu/np.linalg.norm(v_neu+1e-12)) @ (axis/np.linalg.norm(axis)+1e-12))
print("[Bias probe after neutralize] ~", neutral_scores)

# -------------------------
# 5) Sentence similarity with each embedding
# -------------------------
def avg_embed(words, getter, dim):
    vecs = [getter[w] for w in words if w in getter]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)

s1 = "movie was good".split()
s2 = "film was great".split()

def cos(a,b): 
    return float(a @ b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12)

v1 = avg_embed(s1, W2V, W2V.vector_size); v2 = avg_embed(s2, W2V, W2V.vector_size)
print("\nCosine[SGNS] =", round(cos(v1,v2), 3))

u1 = avg_embed([w for w in s1 if w in word2id], {id2word[i]:E_sppmi[i] for i in range(V)}, E_sppmi.shape[1])
u2 = avg_embed([w for w in s2 if w in word2id], {id2word[i]:E_sppmi[i] for i in range(V)}, E_sppmi.shape[1])
print("Cosine[SPPMI/SVD] =", round(cos(u1,u2), 3))

f1 = avg_embed(s1, FT, FT.vector_size); f2 = avg_embed(s2, FT, FT.vector_size)
print("Cosine[FastText] =", round(cos(f1,f2), 3))
