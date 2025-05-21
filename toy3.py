import re
import numpy as np
import pandas as pd
from rich import print as rprint
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from tqdm import trange, tqdm 

from rcode import *
from matplotlib import pyplot as plt

import torch
import transformer_lens
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES

torch.set_grad_enabled(False)
from transformers import GenerationConfig

# --
# Helpers

def to_np(x):
    return x.cpu().numpy()

# Strip repeated eos_token from end of string
def strip_repeated_eos(text, eos_token):
    escaped_eos = re.escape(eos_token)
    pattern     = f"({escaped_eos})+$"
    return re.sub(pattern, eos_token, text)

def extract_choice(x):
    pattern = r'<choice>(.*?)</choice>'
    matches = re.findall(pattern, x)
    if len(matches) == 0:
        return 'heads'
    
    return matches[-1] if matches else None

def n_tokens(x):
    # does this handle special tokens correctly?
    return len(model.tokenizer.encode(x))

# --

model_str = "Qwen/Qwen2-7B-Instruct"
generation_config = GenerationConfig.from_pretrained(model_str)

model = transformer_lens.HookedTransformer.from_pretrained(model_str)

model.tokenizer.padding_side = 'left'
model.tokenizer.pad_token    = model.tokenizer.eos_token

# --
# Generation

input = [
    {"role": "system", "content": "You are a helpful assistant.  Please follow all instructions very carefully."},
    {"role": "user",   "content": 'Output a short fact about a US state capitol. Then output one of "heads" or "tails" wrapped in <choice></choice> tags.'}
]

input_str   = model.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
output_toks = model.generate([input_str] * 256, max_new_tokens=128, top_k=generation_config.top_k, top_p=generation_config.top_p, temperature=generation_config.temperature, return_type='tokens')

output = [model.tokenizer.decode(o, skip_special_tokens=False) for o in output_toks]
output = [strip_repeated_eos(o, model.tokenizer.eos_token) for o in output]
output = list(set(output)) # deduplicate

drop   = np.array([n_tokens(xx) for xx in output]) - n_tokens(input_str) < 30
output = [o for i, o in enumerate(output) if not drop[i]]

# --
# Get probabilities 

padding_side = 'left'

# torch.cuda.empty_cache()
logits, activations = model.run_with_cache(output, padding_side=padding_side, names_filter=lambda hook_name: 'resid_post' in hook_name)
activations         = activations.to('cpu')
logits              = logits.to('cpu')

assert padding_side == 'left'
# model.tokenizer.convert_ids_to_tokens(logits[:,-6].argmax(axis=-1))
choice_idxs  = model.tokenizer.encode(['heads', 'tails'])
Y            = logits[:,-6].softmax(axis=-1)[:,choice_idxs]

drop = Y.sum(axis=-1) < 0.9

output = [o for i, o in enumerate(output) if not drop[i]]
Y      = Y[~drop]
z      = (Y[:,0] / (1 - Y[:,0])).log()

# --
# Train simple classifier

padding_side = 'right'

# torch.cuda.empty_cache()
model.reset_hooks()
logits, activations = model.run_with_cache(output, padding_side=padding_side, names_filter=lambda hook_name: 'resid_post' in hook_name)
activations         = activations.to('cpu')
logits              = logits.to('cpu')

if padding_side == 'left':
    token_idxs = sorted(list(range(-1, -32, -1)))
else:
    token_idxs = sorted(list(range(n_tokens(input_str) - 5, n_tokens(input_str) + 16, 1)))

pd.DataFrame([np.array(model.tokenizer.tokenize(xx))[token_idxs] for xx in output])

out = []
for it in trange(32):
    for layer_idx in [-2]:
        layer_id = list(activations.keys())[layer_idx]
        for token_idx in token_idxs:
            
            X = activations[layer_id][:,token_idx].numpy()
            X = X / np.sqrt((X ** 2).sum(axis=-1, keepdims=True))
            Xp, zp = X[p], z[p]
            
            p      = np.random.RandomState(it).permutation(len(X))
            z_pred = cross_val_predict(Ridge(alpha=0.1), Xp, zp)
            
            stat, pval = spearmanr(zp, z_pred)
            out.append({"token_idx": token_idx, "layer_idx": layer_idx, "stat": float(stat), "pval": float(pval), "it": it})
            print(out[-1])


df_out = pd.DataFrame(out)
for it in df_out.it.unique():
    sub = df_out[df_out['it'] == it]
    _   = plt.plot(sub['token_idx'], sub['stat'], label=f'it {it}', alpha=0.25, c='grey')

_ = plt.plot(df_out.groupby('token_idx').stat.mean(), label='mean', c='black')

_ = plt.grid('both', c='grey', alpha=0.5)
_ = plt.xlabel('token index')
# _ = plt.xlim(-10, 0)
show_plot()


plt.scatter(z_valid, clf.predict(X_valid))
show_plot()

# # --

# token_idx = -15 # left
token_idx = n_tokens(input_str) + 15 # right
layer_idx = -5
layer_id  = list(activations.keys())[layer_idx]

X   = activations[layer_id][:,token_idx].numpy()
X   = X / np.sqrt((X ** 2).sum(axis=-1, keepdims=True))
clf = Ridge(alpha=0.1).fit(X, z)

z_pred = cross_val_predict(Ridge(alpha=0.1), X, z)
spearmanr(z, z_pred)

model.reset_hooks()

traj = []
def my_hook(act, hook):
    tmp = to_np(act[:,-1])
    tmp = tmp / np.sqrt((tmp ** 2).sum(axis=-1, keepdims=True))
    traj.append(clf.predict(tmp))
    return act

model.add_hook(layer_id, my_hook)
_output_toks = model.generate([input_str] * 128, max_new_tokens=128, top_k=generation_config.top_k, top_p=generation_config.top_p, temperature=generation_config.temperature, return_type='tokens')

traj = np.vstack(traj).T
_output_toks = _output_toks[:,n_tokens(input_str):]
_output_toks = to_np(_output_toks)

# eos_idx = model.tokenizer.encode(model.tokenizer.eos_token)[0]
# scores  = [float(t[toks != eos_idx][-token_idx]) for t, toks in zip(traj, _output_toks)]
# scores  = np.array(scores)

scores = traj[:,token_idx - n_tokens(input_str) + 1]

_output = [model.tokenizer.decode(o, skip_special_tokens=False) for o in _output_toks]
_output = [strip_repeated_eos(o, model.tokenizer.eos_token) for o in _output]

y = np.array([extract_choice(o) == 'heads' for o in _output])

metrics.roc_auc_score(y, scores)

for tt, yy in zip(traj, y):
    _ = plt.plot(tt[10:30], alpha=0.25, c='red' if yy else 'blue')

show_plot()