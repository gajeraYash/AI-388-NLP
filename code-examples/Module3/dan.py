# Deep Averaging Network (DAN) for toy sentiment classification
# Avg(embeddings) -> MLP -> Softmax. With optional word dropout.
# References: lecture segment on applying embeddings with DAN; Iyyer et al. 2015.

# pip install torch==2.3.1
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random

# -----------------------------
# 1) Tiny labeled corpus
# -----------------------------
raw = [
    # Positive
    ("the movie was good and fun", 1),
    ("the film was great and enjoyable", 1),
    ("absolutely fantastic film", 1),
    ("i really loved this movie", 1),
    ("the acting was brilliant and the story touching", 1),
    ("a masterpiece with stunning visuals", 1),
    ("the soundtrack was amazing", 1),
    ("a delightful experience overall", 1),
    ("i liked the humor and the pacing", 1),
    ("not bad at all", 1),   # negation flips
    ("this was surprisingly good", 1),
    ("the plot was engaging and full of twists", 1),
    ("wonderful cinematography and acting", 1),
    ("she delivered a powerful performance", 1),
    ("i enjoyed every moment of it", 1),

    # Negative
    ("the plot was bad and boring", 0),
    ("terrible acting and poor script", 0),
    ("i disliked this film", 0),
    ("the pacing was awful", 0),
    ("not good at all", 0),
    ("the dialogue felt unnatural", 0),
    ("a disappointing sequel", 0),
    ("completely unwatchable mess", 0),
    ("the effects were cheap and distracting", 0),
    ("i regret watching this", 0),
    ("the story was flat and predictable", 0),
    ("this was a waste of time", 0),
    ("a poorly written film", 0),
    ("he gave a weak performance", 0),

    # Tricky / Mixed
    ("the movie was not good", 0),   # negation
    ("the movie was not bad", 1),    # double negation → positive
    ("the movie was good but too long", 1),
    ("the movie was bad but had some funny parts", 0),
    ("it was okay, not the best but not the worst", 1),
    ("some scenes were dull but overall enjoyable", 1),
    ("great visuals but terrible story", 0),
    ("mediocre film with a few highlights", 0),
    ("the film was confusing yet interesting", 0),

    # Neutral / borderline
    ("the film was released in 2020", 0),
    ("actors wore blue costumes", 0),
    ("this movie is two hours long", 0),
]
texts, labels = zip(*raw)


# -----------------------------
# 2) Tokenize + vocab
# -----------------------------
def tok(s): return s.lower().split()
tokens = [w for s in texts for w in tok(s)]
vocab = ["<PAD>", "<UNK>"] + sorted({w for w in tokens})
stoi = {w:i for i,w in enumerate(vocab)}

def encode(s):
    return [stoi.get(w, stoi["<UNK>"]) for w in tok(s)]

encoded = [encode(s) for s in texts]
y = torch.tensor(labels, dtype=torch.long)

# -----------------------------
# 3) Dataset with simple padding
# -----------------------------
def pad_to(xs, L, pad=0):
    return xs + [pad]*(L-len(xs))

maxlen = max(len(x) for x in encoded)

class SentData(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): 
        return torch.tensor(pad_to(self.X[i], maxlen)), self.y[i]

ds = SentData(encoded, y)
dl = DataLoader(ds, batch_size=4, shuffle=True)

# -----------------------------
# 4) DAN model
# -----------------------------
class DAN(nn.Module):
    def __init__(self, vocab_size, d_embed=100, hidden=(128, 64), nclass=2, p_worddrop=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_embed, padding_idx=0)
        self.p_worddrop = p_worddrop
        layers = []
        in_dim = d_embed
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, nclass)

    def forward(self, x):
        # x: [B, T]
        E = self.emb(x)  # [B, T, D]

        if self.training and self.p_worddrop > 0:
            # token-level dropout mask
            mask_bt = (torch.rand(x.size(0), x.size(1), device=x.device) > self.p_worddrop)  # [B,T] bool
            mask = mask_bt.unsqueeze(-1).float()  # [B,T,1]
            E = E * mask

            # ensure at least one token kept per sequence
            keep_batch = (mask.sum(dim=1) == 0).squeeze(-1)  # [B] bool
            if keep_batch.any():
                keep_idx = keep_batch.nonzero(as_tuple=True)[0]        # [K]
                E[keep_idx, 0, :] = self.emb(x[keep_idx, 0])           # set first token back
                mask[keep_idx, 0, 0] = 1.0

            denom = mask.sum(dim=1).clamp_min(1.0)  # [B,1]
        else:
            # use exact length as denominator at eval time
            denom = torch.full((x.size(0), 1), x.size(1), dtype=E.dtype, device=E.device)

        z = E.sum(dim=1) / denom           # [B,D] average embeddings
        h = self.mlp(z)
        return self.out(h)

# -----------------------------
# 5) Train
# -----------------------------
device = "cpu"
model = DAN(vocab_size=len(vocab), d_embed=100, hidden=(128,64), nclass=2, p_worddrop=0.3).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
crit = nn.CrossEntropyLoss()

for epoch in range(25):
    model.train()
    running = 0.0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward()
        opt.step()
        running += loss.item()*xb.size(0)
    avg_loss = running/len(ds)

# -----------------------------
# 6) Inference helper
# -----------------------------
model.eval()
def predict(text):
    with torch.no_grad():
        ids = torch.tensor([pad_to(encode(text), maxlen)])
        logits = model(ids)
        prob = F.softmax(logits, dim=-1).squeeze(0)
        label = int(prob.argmax().item())
        return {"text": text, "probs": prob.tolist(), "label": ["neg","pos"][label]}

tests = [
    "the movie was good",
    "the film was great",
    "the movie was bad",
    "the movie was not bad",
    "awful script but funny moments",
]
for s in tests:
    print(predict(s))
