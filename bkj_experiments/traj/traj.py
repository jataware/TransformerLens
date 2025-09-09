
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/openai/protocol.py#L720
# _DEFAULT_SAMPLING_PARAMS = {
#     "temperature": 0.7,
#     "top_p": 1.0,
#     "top_k": -1,
#     "min_p": 0.0,
#     "repetition_penalty": 1.0,
# }
# ??? Is this right?
    
from matplotlib import pyplot as plt
from rcode import *
from seaborn import heatmap

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from rich import print as rprint
from tqdm import trange
from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import torch
torch.set_grad_enabled(False)
from torch.nn.functional import normalize

from inspect_model import InspectModel

__here__ = Path(__file__).parent

ins = InspectModel(padding_side='right')

# --
# IO

inpaths = [
    Path(__here__) / 'data/seed_10016.jl',
    Path(__here__) / 'data/seed_10116.jl',
    Path(__here__) / 'data/seed_10232.jl',
]

data   = [json.loads(line) for inpath in inpaths for line in inpath.open('r').readlines()]
df     = pd.DataFrame(data)


# --
# Clean

# Add question IDs
q2idx       = {q: i for i, q in enumerate(df.question.unique())}
df['_qidx'] = df.question.map(q2idx)
df          = df.sort_values('_qidx').reset_index(drop=True)
df['_iter'] = df.groupby('_qidx').question.transform(lambda x: range(len(x)))

# Get actual answers
def letter2str(df, col):
    letter2idx = {letter: i for i, letter in enumerate('ABCD')}
    idxs       = df[col].map(lambda x: letter2idx.get(x, -1)).values
    
    _strs = []
    for choices, aidx in zip(df.choices.values, idxs):
        _strs.append(choices[aidx] if aidx != -1 else None)
    
    return _strs


df['answer_str']  = letter2str(df, 'answer')
df['target_str']  = letter2str(df, 'target')

assert (df.correct == (df.answer_str == df.target_str)).all()
assert df.groupby('_qidx').target_str.apply(lambda x: 1 == len(set(x))).values.all()

# # --
# # sanity checks

# does majority voting help?
# yes ... though not be a ton
df.correct.mean()
df.groupby('_qidx').apply(lambda x: Counter(x.answer_str).most_common(1)[0][0] == x.target_str.values[0]).mean()

# is consistency correlated with correctness?
# yes
df["_maj"] = df.groupby('_qidx').answer_str.transform(lambda x: Counter(x).most_common(1)[0][0])
df["_n"]   = df.groupby('_qidx').answer_str.transform(lambda x: len(set(x)))

udf = df.drop_duplicates('_qidx')
pd.crosstab(udf._n, udf._maj == udf.target_str, normalize='index')

# [TODO] consistency is highly correlated with accuracy ...

df['c'] = df.answer_str == df._maj

# # pct of responses in majority class is also correlated with accuracy ... maybe this is better
# # than just unique counts
a = df.groupby('_qidx').c.mean().values
b = df.groupby('_qidx').apply(lambda x: x._maj.values[0] == x.target_str.values[0]).values
z = pd.Series(b).groupby(a.round(1)).mean()
_ = plt.plot(z.index, z.values)
show_plot()

pd.crosstab(a.round(1), b)
pd.crosstab(a.round(1), b, normalize='index')

# # but then the question is how to estimate "pct of responses in majority class" efficiently

# # [TODO] demonstrate this is actually useful, w/ a large number of samples.  does it get
# #        meaningfully better as you go from 8, 16, 32, ..., 128 samples?
# #        - confounder: temperature, ...
# # [IDEA] train a model to predict class from partial rollouts and then use that compute % majority
# #        class for confidence estimation
# # [IDEA] do some unsupervised thing on partial embeddings to get confidence estimates
# # [IDEA] can you detect which one is going to be the majority vote by looking at the trajectory?
# # >>

# --
# Pick a question

n_answers = df.groupby('_qidx').answer_str.apply(lambda x: len(set(x)))
keep      = n_answers.index[n_answers.values == 2]

sub       = df[df._qidx.isin(keep)].groupby('_qidx').correct.mean()
sub       = sub[(sub > 0.25) & (sub < 0.75)]

qidx      = sub.index[4]
sub       = df[df['_qidx'] == qidx]

uanswer_str = sub.answer_str.unique()
answer2idx  = {a: i for i, a in enumerate(uanswer_str)}
y           = np.array([answer2idx[a] for a in sub.answer_str.values])
assert len(set(y)) == 2 # [TODO] fix this

breakpoint()

# --
# Compute activations

SYSTEM_PROMPT = {"role": "system", "content": "You are a helpful assistant. /no_think"}
messages      = [
    [
        SYSTEM_PROMPT, 
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": output_str},
    ] for prompt, output_str in sub[['prompt', 'output_str']].values
]

def _layer_filter(layer_name):
    pattern = re.compile(r'blocks\.(\d+)\.hook_resid_post')
    match   = pattern.match(layer_name)
    if match:
        return int(match.group(1)) >= 25
    
    return False

try:
    del cache
except:
    pass

cache         = ins.batched_forward(messages, tokens_per_batch=8192, names_filter=_layer_filter)
prefix_tokens = ins.n_tokens(ins.prep(messages[0][:2]))
output_tokens = [ins.n_tokens(ins.prep([message[-1]])) for message in messages]


# drop prefix
cache = [
    {k: v[prefix_tokens:] for k, v in c.items()}
    for c in cache
]

# --
# Train model
# This seems to be doing something ...

from tqdm import trange
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

# toks = [ins.model.tokenizer.encode(ins.prep(m)) for m in messages]
# X = TfidfVectorizer().fit_transform([' '.join([str(xxx) for xxx in xx]) for xx in toks])

def run_one(seed, acts, output_tokens, y, n_train=2, p_toks=None, n_toks=None, extra=None):
    if extra is None:
        extra = {}
    
    rng = np.random.RandomState(seed)
    assert p_toks is not None or n_toks is not None
    
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    
    sel       = np.hstack([rng.choice(idx0, n_train , replace=False), rng.choice(idx1, n_train, replace=False)])
    train_sel = np.isin(np.arange(len(y)), sel)
    valid_sel = ~train_sel
    
    if n_toks is not None:
        mean_acts = np.array([a[:n_toks].mean(axis=0) for a in acts])
    elif p_toks is not None:
        mean_acts = np.array([a[:int(t * p_toks)].mean(axis=0) for a, t in zip(acts, output_tokens)])
    
    clf   = LogisticRegression(max_iter=10000, random_state=seed)
    clf   = clf.fit(mean_acts[train_sel], y[train_sel])
    y_hat = clf.predict_proba(mean_acts[valid_sel])[:,1]
    
    return {
        "n_train" : n_train,
        "p_toks"  : p_toks,
        "n_toks"  : n_toks,
        "roc_auc" : roc_auc_score(y[valid_sel], y_hat),
        **extra
    }

ptype = 'n_toks'

jobs = []
n_replicates = 64
for layer in [30]:
    acts = [c[f'blocks.{layer}.hook_resid_post'][:,prefix_tokens:] for c in cache]
    acts = [a.clone().numpy() for a in acts]
    
    for n_train in [2, 4, 8, 16]:
        # for psize in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        for psize in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
            run_kwargs = {
                "n_train" : n_train,
                ptype     : psize,
                "extra"   : {"layer": layer}
            }
            
            jobs += [
                delayed(run_one)(_,acts, output_tokens, y, **run_kwargs)
                for _ in range(n_replicates)
            ]

res    = Parallel(n_jobs=-1, verbose=-1)(jobs)
df_res = pd.DataFrame(res)

# Create a table with mean ROC AUC scores
tab = df_res.groupby(['n_train', ptype, 'layer'])[['roc_auc']].mean().reset_index()
# Create grid of plots - one per layer

layers = sorted(tab['layer'].unique())
fig, axes = plt.subplots(1, len(layers), figsize=(7.5 * len(layers), 5))

# Calculate global y-axis limits
y_min = 0
y_max = 1

for i, layer in enumerate(layers):
    ax = axes[i] if len(layers) > 1 else axes
    layer_data = tab[tab['layer'] == layer]
    
    # Plot one line per n_train value
    for cidx, n_train in enumerate(sorted(layer_data['n_train'].unique())):
        n_train_data = layer_data[layer_data['n_train'] == n_train]
        
        _ = ax.plot(n_train_data[ptype], n_train_data['roc_auc'], 
                marker='o', label=f'n_train={n_train}', c=f'C{cidx}')
        
    _ = ax.set_xlabel(ptype)
    _ = ax.set_ylabel('ROC AUC')
    _ = ax.set_title(f'Layer {layer}')
    _ = ax.legend()
    _ = ax.grid(True, alpha=0.3)
    _ = ax.set_ylim(y_min, y_max)

_ = plt.tight_layout()
_ = plt.savefig('roc_auc.png')
