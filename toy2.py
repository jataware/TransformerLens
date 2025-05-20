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

def strip_repeated_eos(text, eos_token):
    escaped_eos = re.escape(eos_token)
    pattern     = f"({escaped_eos})+$"
    return re.sub(pattern, eos_token, text)

# --
# Fact generation

input = [
    {"role": "system", "content": "You are a helpful assistant.  Please follow all instructions very carefully."},
    {"role": "user",   "content": 'Output a short fact about a US state capitol.'}
]

input_str   = model.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
output_toks = model.generate([input_str] * 128, max_new_tokens=128, top_k=generation_config.top_k, top_p=generation_config.top_p, temperature=generation_config.temperature, return_type='tokens')

output = [model.tokenizer.decode(o, skip_special_tokens=False) for o in output_toks]
output = [strip_repeated_eos(o, model.tokenizer.eos_token) for o in output]
output = list(set(output)) # deduplicate

output = [o.replace(input_str, '') for o in output]
output = ['<|im_end|>'.join(o.split('<|im_end|>')[:-1]) for o in output]
facts  = output

# --
# Heads / tails generation

input = [
    {"role": "system", "content": "You are a helpful assistant.  Please follow all instructions very carefully."},
    {"role": "user",   "content": 'Output a short fact about a US state capitol. Then output one of "heads" or "tails" wrapped in <choice></choice> tags.'}
]

input_str  = model.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
input_strs = [input_str + o + '\n <choice>' for o in facts]

logits      = model(input_strs)
last_logits = logits[:,-1]

idxs        = model.tokenizer.encode(['heads', 'tails'])
probs       = last_logits.softmax(axis=-1)[:,idxs]

_ = plt.plot(np.sort(to_np(probs[:,0])))
_ = plt.xlabel('Rank')
_ = plt.ylabel('Probability of heads')
show_plot()

ranks     = np.argsort(probs[:,0].cpu().numpy())
output_lo = model.generate([input_strs[ranks[0]]] * 100, max_new_tokens=128, top_k=generation_config.top_k, top_p=generation_config.top_p, temperature=generation_config.temperature)
output_hi = model.generate([input_strs[ranks[-1]]] * 100, max_new_tokens=128, top_k=generation_config.top_k, top_p=generation_config.top_p, temperature=generation_config.temperature)

(np.array([extract_choice(o) for o in output_lo]) == 'heads').mean()
# 0.1

(np.array([extract_choice(o) for o in output_hi]) == 'heads').mean()
# 1.0

# --
# Get activations

n = 40

input_lo = [input_strs[i] for i in ranks[:n]]
log_lo, act_lo = model.run_with_cache(input_lo, names_filter=lambda hook_name: 'resid_mid' in hook_name,)
log_lo, act_lo = log_lo.to('cpu'), act_lo.to('cpu')

input_hi = [input_strs[i] for i in ranks[-n:]]
log_hi, act_hi = model.run_with_cache(input_hi, names_filter=lambda hook_name: 'resid_mid' in hook_name)
log_hi, act_hi = log_hi.to('cpu'), act_hi.to('cpu')

# --
# Train classifier

np.sort([len(model.tokenizer.encode(o)) for o in facts])

layer_idx = -5
layer_id  = list(act_lo.keys())[layer_idx]

out = []
for it in trange(32):
    idx_train, idx_valid = train_test_split(range(2 * n), test_size=0.2)
    
    for token_idx in range(-1, -20, -1):
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
