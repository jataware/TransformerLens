import re
import numpy as np
import pandas as pd
from rich import print as rprint
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from tqdm import trange, tqdm 

from rcode import *
from matplotlib import pyplot as plt

import torch
import transformer_lens
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES

torch.set_grad_enabled(False)
from transformers import GenerationConfig

model_str = "Qwen/Qwen2-7B-Instruct"
generation_config = GenerationConfig.from_pretrained(model_str)

model = transformer_lens.HookedTransformer.from_pretrained(model_str)

model.tokenizer.padding_side = 'left'
model.tokenizer.pad_token    = model.tokenizer.eos_token

# --
# Helpers

def to_np(x):
    return x.cpu().numpy()

def extract_choice(x):
    pattern = r'<choice>(.+)</choice>'
    matches = re.findall(pattern, x)
    return matches[-1] if matches else None

def strip_eos(text, eos_token):
    escaped_eos = re.escape(eos_token)
    pattern     = f"({escaped_eos})+$"
    return re.sub(pattern, '', text)

def n_tokens(x):
    # does this handle special tokens correctly?
    return len(model.tokenizer.encode(x))

def first_n_tokens(x, n):
    return model.tokenizer.decode(model.tokenizer.encode(x)[:n])

# --
# Fact generation
# !! Could probably get better facts w/ gemini or something, though that may be 

fact_msgs = [
    {"role": "system", "content": "You are a helpful assistant.  Please follow all instructions very carefully."},
    {"role": "user",   "content": 'Output a short fact about a US state capitol.'}
]

fact_prompt   = model.tokenizer.apply_chat_template(fact_msgs, tokenize=False, add_generation_prompt=True)
fact_toks_out = model.generate([fact_prompt] * 512, max_new_tokens=128, top_k=generation_config.top_k, top_p=generation_config.top_p, temperature=generation_config.temperature, return_type='tokens')

fact_strs_out = [model.tokenizer.decode(xx, skip_special_tokens=False) for xx in fact_toks_out] # convert to string
fact_strs_out = [xx.replace(fact_prompt, '') for xx in fact_strs_out] # strip prompt
fact_strs_out = [strip_eos(o, model.tokenizer.eos_token) for o in fact_strs_out] # strip eos
fact_strs_out = list(set(fact_strs_out)) # deduplicate

# --
# Heads / tails generation

choice_msgs = [
    {"role": "system", "content": "You are a helpful assistant.  Please follow all instructions very carefully."},
    {"role": "user",   "content": 'Output a short fact about a US state capitol. Then output one of "heads" or "tails" wrapped in <choice></choice> tags.'}
]

choice_prompt  = model.tokenizer.apply_chat_template(choice_msgs, tokenize=False, add_generation_prompt=True)
choice_prompts = [choice_prompt + fact_str + '\n<choice>' for fact_str in fact_strs_out]

logits = model(choice_prompts)[:, -1]
probs  = logits.softmax(axis=-1)

choice_idxs  = model.tokenizer.encode(['heads', 'tails'])
choice_probs = probs[:,choice_idxs]
p_heads      = to_np(choice_probs[:,0])
ranks        = np.argsort(p_heads)

_ = plt.plot(np.sort(p_heads))
_ = plt.xlabel('Prompt Rank')
_ = plt.ylabel('Probability of "heads"')
_ = plt.grid('both', c='grey', alpha=0.5)
show_plot()

# sanity check ... this is obviously correct though
output_lo = model.generate([choice_prompts[ranks[0]]] * 100, max_new_tokens=128, top_k=generation_config.top_k, top_p=generation_config.top_p, temperature=generation_config.temperature)
output_hi = model.generate([choice_prompts[ranks[-1]]] * 100, max_new_tokens=128, top_k=generation_config.top_k, top_p=generation_config.top_p, temperature=generation_config.temperature)

(np.array([extract_choice(o) for o in output_lo]) == 'heads').mean()
# 0.1

(np.array([extract_choice(o) for o in output_hi]) == 'heads').mean()
# 1.0

# --
# Get activations from top / bottom n prompts

n = 50
padding_side = 'right'

input_lo  = [choice_prompts[i] for i in ranks[:n]]
_, act_lo = model.run_with_cache(input_lo, padding_side=padding_side, names_filter=lambda hook_name: 'resid_mid' in hook_name,)
act_lo    = act_lo.to('cpu')

input_hi  = [choice_prompts[i] for i in ranks[-n:]]
_, act_hi = model.run_with_cache(input_hi, padding_side=padding_side, names_filter=lambda hook_name: 'resid_mid' in hook_name)
act_hi    = act_hi.to('cpu')

# --
# Train classifier

layer_idx = -5
layer_id  = list(act_lo.keys())[layer_idx]

token_ids = list(range(n_tokens(choice_prompt), n_tokens(choice_prompt) + 10, 1))

out = []
for it in trange(32):
    idx_train, idx_valid = train_test_split(range(2 * n), test_size=0.2)
    
    for token_idx in token_ids:
        X_hi = act_hi[layer_id][:,token_idx].numpy()
        X_lo = act_lo[layer_id][:,token_idx].numpy()
        
        X    = np.concatenate([X_hi, X_lo], axis=0)
        X    = X / np.sqrt((X ** 2).sum(axis=-1, keepdims=True))
        
        y    = np.concatenate([np.ones(len(X_hi)), np.zeros(len(X_lo))])
        
        X_train, X_valid, y_train, y_valid = X[idx_train], X[idx_valid], y[idx_train], y[idx_valid]
        
        clf = LinearSVC(max_iter=10000).fit(X_train, y_train)
        acc  = metrics.roc_auc_score(y_valid, clf.decision_function(X_valid))
        out.append({"token_idx": token_idx, "layer_idx": layer_idx, "acc": acc, "it": it})
        print(out[-1])

df_out = pd.DataFrame(out)
for it in df_out.it.unique():
    sub = df_out[df_out['it'] == it]
    _   = plt.plot(sub['token_idx'], sub['acc'], label=f'it {it}', alpha=0.25, c='grey')

_ = plt.plot(df_out.groupby('token_idx').acc.mean(), label='mean', c='black')

_ = plt.grid('both', c='grey', alpha=0.5)
_ = plt.xlabel('token index')
_ = plt.ylabel('ROC AUC')
show_plot()

# Upshot: can get pretty good classifier for outcome looking at the first ~5 tokens

# --
# What if we train a single classifier on all tokens?

X_hi = act_hi[layer_id][:,n_tokens(choice_prompt)+3:n_tokens(choice_prompt)+13]
X_lo = act_lo[layer_id][:,n_tokens(choice_prompt)+3:n_tokens(choice_prompt)+13]

X_hi = X_hi / np.sqrt((X_hi ** 2).sum(axis=-1, keepdims=True))
X_lo = X_lo / np.sqrt((X_lo ** 2).sum(axis=-1, keepdims=True))

X_hi_train, X_hi_valid = train_test_split(X_hi, test_size=0.2)
X_lo_train, X_lo_valid = train_test_split(X_lo, test_size=0.2)

Z_hi_train = X_hi_train.reshape(-1, 3584)
Z_hi_valid = X_hi_valid.reshape(-1, 3584)
Z_lo_train = X_lo_train.reshape(-1, 3584)
Z_lo_valid = X_lo_valid.reshape(-1, 3584)

Z_train = np.concatenate([Z_hi_train, Z_lo_train], axis=0)
Z_valid = np.concatenate([Z_hi_valid, Z_lo_valid], axis=0)

y_train = np.concatenate([np.ones(len(Z_hi_train)), np.zeros(len(Z_lo_train))])
y_valid = np.concatenate([np.ones(len(Z_hi_valid)), np.zeros(len(Z_lo_valid))])

clf = LinearSVC(max_iter=10000).fit(Z_train, y_train)
acc = metrics.roc_auc_score(y_valid, clf.decision_function(Z_valid))
print(acc)
# > .80 ... need more samples to really train


scores_lo = clf.decision_function(X_lo_valid.reshape(-1, 3584)).reshape(len(X_lo_valid), -1)
scores_hi = clf.decision_function(X_hi_valid.reshape(-1, 3584)).reshape(len(X_hi_valid), -1)

# What if we apply the classifier rolling forward?
for xx in scores_lo:
    _ = plt.plot(xx, c='blue', alpha=0.25)

for xx in scores_hi:
    _ = plt.plot(xx, c='red', alpha=0.25)

_ = plt.plot(scores_lo.mean(axis=0), c='blue')
_ = plt.plot(scores_hi.mean(axis=0), c='red')

show_plot()

# --

traj = []
def my_hook(act, hook):
    global traj
    tmp = to_np(act[:,-1])
    tmp = tmp / np.sqrt((tmp ** 2).sum(axis=-1, keepdims=True))
    traj.append(clf.decision_function(tmp))
    return act

model.reset_hooks()
model.add_hook(layer_id, my_hook)

output = model.generate([choice_prompt] * 128, max_new_tokens=128, top_k=generation_config.top_k, top_p=generation_config.top_p, temperature=generation_config.temperature)
traj   = np.row_stack(traj).T

model.reset_hooks()
tmp     = ['<choice>'.join(xx.split('<choice>')[:-1]) + '<choice>' for xx in output]
_logits = model(tmp)[:, -1]
_probs  = _logits.softmax(axis=-1)
_choice_probs = _probs[:,choice_idxs]
_p_heads      = to_np(_choice_probs[:,0])

_ = plt.plot(np.sort(_p_heads))
show_plot()


plt.scatter(traj[:,10], _p_heads)
show_plot()

from scipy.stats import spearmanr
spearmanr(traj[:,10], _p_heads)


z = np.array([extract_choice(o) for o in output]) == 'heads'

for xx, zz in zip(traj, z):
    _ = plt.plot(xx[:20], c='red' if zz else 'blue', alpha=0.25)

show_plot()

_ = plt.plot([metrics.roc_auc_score(z, traj[:,i]) for i in range(traj.shape[1])])
show_plot()

_ = plt.plot(traj[z].mean(axis=0)[:20], c='red')
_ = plt.plot(traj[~z].mean(axis=0)[:20], c='blue')
show_plot()

_ = plt.hist(traj[z,10], bins=100)
_ = plt.hist(traj[~z,10], bins=100)
show_plot()
