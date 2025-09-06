
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

import transformer_lens

__here__ = Path(__file__).parent


class InspectModel:
    def __init__(self, model_str='Qwen/Qwen3-8B', padding_side='right'):
        self.model_str                    = model_str
        self.model                        = transformer_lens.HookedTransformer.from_pretrained(model_str)
        self.model.tokenizer.pad_token    = self.model.tokenizer.eos_token
        
        self.model.tokenizer.padding_side = padding_side
    
    def n_tokens(self, x):
        return len(self.model.tokenizer.encode(x))
    
    def prep(self, messages):
        return self.model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    def forward(self, messages, chunk_size=16):
        # <<
        self.model.reset_hooks()
        logits, activations = self.model.run_with_cache(
            self.prep(messages), 
            padding_side = self.model.tokenizer.padding_side, 
            names_filter = lambda hook_name: 'resid_post' in hook_name # Is this right?
        )
        logits      = logits.to('cpu')
        activations = activations.to('cpu')
        
        return logits, activations
        # --        
        # all_logits      = []
        # all_activations = []
        # for offset in range(0, len(messages), chunk_size):
        #     self.model.reset_hooks()
        #     messages_chunk = messages[offset:offset+chunk_size]
        #     print(f'{len(messages_chunk)} messages')
        #     logits, activations = self.model.run_with_cache(
        #         self.prep(messages_chunk), 
        #         padding_side = self.model.tokenizer.padding_side, 
        #         names_filter = lambda hook_name: 'resid_post' in hook_name # Is this right?
        #     )
        #     all_logits.append(logits.to('cpu'))
        #     all_activations.append(activations.to('cpu'))
        
        # # breakpoint() # [TODO] get the batch to work
        # return all_logits, all_activations
        # >>


ins = InspectModel()

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

acc  = df.groupby('_qidx').correct.mean()
acc  = acc[(acc > 0.4) & (acc < 0.6)]
qidx = acc.index[3]

sub  = df[df['_qidx'] == qidx]


uanswer_str = sub.answer_str.unique()
answer2idx  = {a: i for i, a in enumerate(uanswer_str)}
y           = np.array([answer2idx[a] for a in sub.answer_str.values])

assert len(set(y)) == 2 # [TODO] fix this

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

logits, activations = ins.forward(messages)

prefix_tokens = ins.n_tokens(ins.prep(messages[0][:2]))
output_tokens = [ins.n_tokens(ins.prep([message[-1]])) for message in messages]

acts = activations['blocks.25.hook_resid_post'][:,prefix_tokens:].clone()
acts = acts.numpy()

# breakpoint()

# # # <<
# from scipy.spatial.distance import pdist, squareform
# mean_acts = np.array([a[:int(t * 0.1)].mean(axis=0) for a, t in zip(acts, output_tokens)])

# mean_acts = mean_acts[np.argsort(y)]

# dist      = squareform(pdist(np.sign(mean_acts) * np.sqrt(np.abs(mean_acts)), metric='cosine'))
# _         = heatmap(dist, cmap='viridis')
# show_plot()
# # # >>


# --
# Train model
# This seems to be doing something ...

# [TODO] use multiple layers
# [TODO] use different aggregations
# [TODO] preprocessing - there are maybe weird spikes here? clipping? normalizing?

from tqdm import trange
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

def run_one(acts, output_tokens, y, n_train=2, p=1):
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    
    sel       = np.hstack([np.random.choice(idx0, n_train, replace=False), np.random.choice(idx1, n_train, replace=False)])
    train_sel = np.isin(np.arange(len(y)), sel)
    valid_sel = ~train_sel
    
    mean_acts = np.array([a[:int(t * p)].mean(axis=0) for a, t in zip(acts, output_tokens)])
    # mean_acts = mean_acts - mean_acts.mean(axis=0)
    # mean_acts = mean_acts / np.std(mean_acts, axis=0)
    
    clf   = LogisticRegression(max_iter=10000).fit(mean_acts[train_sel], y[train_sel])
    y_hat = clf.predict_proba(mean_acts[valid_sel])[:,1]
    return roc_auc_score(y[valid_sel], y_hat), y_hat.sum(), y[valid_sel].sum()


for layer in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
    acts = activations[f'blocks.{layer}.hook_resid_post'][:,prefix_tokens:].clone()
    acts = acts.numpy()
    
    for n_train in [2]:
        for p in [1]:
            tmp, a, b = zip(*[run_one(acts, output_tokens, y, n_train=n_train, p=p) for _ in range(64)])
            print(layer, n_train, p, np.mean(tmp), np.median(tmp), np.mean(a), np.mean(b))
            _ = plt.plot(np.sort(tmp), label=f'{layer} {n_train} {p}')


_ = plt.axhline(0.5, c='red')
_ = plt.legend(loc='lower right')
show_plot()





# def get_prefix(message, n_tokens):
#     toks = ins.model.tokenizer.apply_chat_template([message], tokenize=True, add_generation_prompt=False, max_length=n_tokens, truncation=True)
#     return ins.model.tokenizer.decode(toks)

# rprint([get_prefix(m[-1], 32) for m in messages])