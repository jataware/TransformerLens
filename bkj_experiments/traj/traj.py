
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
    
    def forward(self, messages):
        self.model.reset_hooks()
        
        logits, activations = self.model.run_with_cache(
            self.prep(messages), 
            padding_side = self.model.tokenizer.padding_side, 
            names_filter = lambda hook_name: 'resid_post' in hook_name # Is this right?
        )
        logits              = logits.to('cpu')
        activations         = activations.to('cpu')
        
        return logits, activations


ins = InspectModel()

# --
# IO

inpaths = [
    Path(__here__) / 'data/seed_10016.jl',
    Path(__here__) / 'data/seed_10116.jl',
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

print(df.head())

# --
# sanity checks

# does majority voting help?
# yes ... though not be a ton
df.correct.mean()
df.groupby('_qidx').apply(lambda x: Counter(x.answer_str).most_common(1)[0][0] == x.target_str.values[0]).mean()

df["_maj"] = df.groupby('_qidx').answer_str.transform(lambda x: Counter(x).most_common(1)[0][0])
df["_n"]   = df.groupby('_qidx').answer_str.transform(lambda x: len(set(x)))

# is consistency correlated with correctness?
# yes
udf = df.drop_duplicates('_qidx')
pd.crosstab(udf._n, udf._maj == udf.target_str, normalize='index')

# --
# Pick a question

acc  = df.groupby('_qidx').correct.mean()
acc  = acc[(acc > 0.6) & (acc < 0.8)]
qidx = acc.index[3]

sub  = df[df['_qidx'] == qidx]

uanswer_str = sub.answer_str.unique()
answer2idx  = {a: i for i, a in enumerate(uanswer_str)}
y           = np.array([answer2idx[a] for a in sub.answer_str.values])

assert len(set(y)) == 2

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

# <<
# prefix_tokens = ins.n_tokens(ins.prep(messages[0][:2]))
# --
prefix_tokens = ins.n_tokens(ins.model.tokenizer.apply_chat_template(messages[0][:2], tokenize=False, add_generation_prompt=False))
output_tokens = [ins.n_tokens(ins.model.tokenizer.apply_chat_template([message[-1]], tokenize=False, add_generation_prompt=False)) for message in messages]
# >>

acts = activations['blocks.30.hook_resid_post'][:,prefix_tokens:].clone()
acts = acts.numpy()

# --
# Train model

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

def rolling_mean(x, window):
    out = pd.DataFrame(x).rolling(window=window).mean().values
    out = out[~np.isnan(out).any(axis=1)]
    return out


def run_one(n_train=2, p=1):
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    
    sel       = np.hstack([np.random.choice(idx0, n_train, replace=False), np.random.choice(idx1, n_train, replace=False)])
    train_sel = np.isin(np.arange(len(y)), sel)
    valid_sel = ~train_sel
    
    mean_acts = np.array([a[:int(t * p)].mean(axis=0) for a, t in zip(acts, output_tokens)])
    
    clf   = LogisticRegression(max_iter=10000).fit(mean_acts[train_sel], y[train_sel])
    y_hat = clf.decision_function(mean_acts[valid_sel])
    return roc_auc_score(y[valid_sel], y_hat)

from tqdm import trange
for n_train in range(1, 3):
    for p in [0.25]:
        tmp = [run_one(n_train=n_train, p=p) for _ in range(64)]
        print(n_train, p, np.mean(tmp), np.median(tmp))
        _ = plt.plot(np.sort(tmp), label=f'{n_train} {p}')

_ = plt.axhline(0.5, c='red')
_ = plt.legend(loc='lower right')
show_plot()





def get_prefix(message, n_tokens):
    toks = ins.model.tokenizer.apply_chat_template([message], tokenize=True, add_generation_prompt=False, max_length=n_tokens, truncation=True)
    return ins.model.tokenizer.decode(toks)

rprint([get_prefix(m[-1], 32) for m in messages])









# # --
# # IO

# traces   = [json.loads(line) for line in open('trace.jl')]
# messages = [
#     [
#         {"role": "system",    "content": "You are a helpful assistant."},
#         {"role": "user",      "content": trace['prompt']},
#         {"role": "assistant", "content": trace['output_str']}
#     ]
#     for trace in traces
# ]




# from sklearn.linear_model import LogisticRegression

# pre_tokens = n_tokens(model.tokenizer.apply_chat_template(messages[0][:2], tokenize=False, add_generation_prompt=False))
# token_cnts = [n_tokens(s) for s in input_str]

# a = activations['blocks.30.hook_resid_post']
# a = a.numpy().copy()
# f = np.vstack([aa[pre_tokens:cc].mean(axis=0) for aa, cc in zip(a, token_cnts)])

# lr = LogisticRegression().fit([f[3], f[2]], [0, 1])

# tmp = []
# for i in range(1, max(token_cnts)):
#     fp = np.vstack([aa[:i].mean(axis=0) for aa, cc in zip(a, token_cnts)])
#     tmp.append(lr.predict_proba(fp)[:,1])

# tmp = np.array(tmp)
# for i, t in enumerate(tmp.T):
#     plt.plot(t[:token_cnts[i]], label=f'{i}')

# _ = plt.legend()
# show_plot()

# # for k in activations.keys():
# #     print(k, activations[k].shape)

# min_tokens = min([n_tokens(s) for s in input_str])



# a = activations['blocks.30.hook_resid_post'].clone()
# a = a[:, pre_tokens:min_tokens]

# f = a[:, -1].mean(axis=1)
# d = squareform(pdist(f, metric='cosine'))
# d.round(2) * 100


# a  = a.numpy()
# an = a
# # an = np.array([pd.DataFrame(x).rolling(window=32).mean().values for x in a])
# an = an / np.sqrt((an ** 2).sum(axis=-1, keepdims=True))

# labs = [0, 1, 1, 0]
# for i in range(an.shape[0]):
#     for j in range(i + 1, an.shape[0]):
#         c = 'red' if labs[i] == labs[j] else 'black'
#         _ = plt.plot(, label=f'{i} vs {j}', c=c)

# _ = plt.legend()
# show_plot()


