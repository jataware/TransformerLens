from matplotlib import pyplot as plt
from rcode import *
from seaborn import heatmap

import json
import numpy as np
import pandas as pd
from pathlib import Path
from rich import print as rprint
from sklearn.linear_model import LogisticRegression
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

inpath = Path(__here__) / 'data/seed_10016.jl'

data   = [json.loads(line) for line in inpath.open('r').readlines()]
df     = pd.DataFrame(data)
df['_seed'] = inpath.stem.split('_')[-1]


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
# df.correct.mean()
# df.groupby('_qidx').apply(lambda x: Counter(x.answer_str).most_common(1)[0][0] == x.target_str.values[0]).mean()
df["_maj"] = df.groupby('_qidx').answer_str.transform(lambda x: Counter(x).most_common(1)[0][0])
df["_n"]   = df.groupby('_qidx').answer_str.transform(lambda x: len(set(x)))

# is consistency correlated with correctness?
# yes
udf = df.drop_duplicates('_qidx')
pd.crosstab(udf._n, udf._maj == udf.target_str, normalize='index')

# --
# Pick a question

qidx = df.groupby('_qidx').correct.mean().sort_values(ascending=False).index[50]
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
# >>

acts = activations['blocks.30.hook_resid_post']
acts = acts[:,prefix_tokens:]


# --
# Train model

from sklearn.svm import LinearSVC

idx0 = np.where(y == 0)[0]
idx1 = np.where(y == 1)[0]

# np.random.seed(123)
sel       = np.hstack([np.random.choice(idx0, 1), np.random.choice(idx1, 1)])
train_sel = np.isin(np.arange(len(y)), sel)
valid_sel = ~train_sel

n_tokens = 64

mean_acts = acts[:,:n_tokens].mean(axis=1)
clf       = LinearSVC().fit(mean_acts[train_sel], y[train_sel])
y_hat     = clf.decision_function(mean_acts[valid_sel])
roc_auc_score(y[valid_sel], y_hat)


def get_prefix(message):
    toks = ins.model.tokenizer.apply_chat_template([message], tokenize=True, add_generation_prompt=False, max_length=n_tokens, truncation=True)
    return ins.model.tokenizer.decode(toks)

rprint([get_prefix(m[-1]) for m in messages])









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


