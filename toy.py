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

# --
# Helpers

# Strip repeated eos_token from end of string
def strip_repeated_eos(text, eos_token):
    escaped_eos = re.escape(eos_token)
    pattern     = f"({escaped_eos})+$"
    return re.sub(pattern, eos_token, text)

def extract_choice(x):
    pattern = r'<choice>(.*?)</choice>'
    matches = re.findall(pattern, x)
    if len(matches) == 0:
        raise Exception()
    
    return matches[-1] if matches else None

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
output_toks = model.generate([input_str] * 128, max_new_tokens=128, top_k=generation_config.top_k, top_p=generation_config.top_p, temperature=generation_config.temperature, return_type='tokens')

output = [model.tokenizer.decode(o, skip_special_tokens=False) for o in output_toks]
output = [strip_repeated_eos(o, model.tokenizer.eos_token) for o in output]
output = list(set(output)) # deduplicate

y  = [extract_choice(xx) for xx in output]
y  = np.array(y) == 'heads'

# --
# Look at activations

logits, activations = model.run_with_cache(output, names_filter=lambda hook_name: 'resid_pre' in hook_name)
activations         = activations.to('cpu')

# --
# Train simple classifier

out = []

for it in trange(32):
    idx_train, idx_valid = train_test_split(range(len(output)), test_size=0.2)
    for layer_idx in [-5]:
        layer_id = list(activations.keys())[layer_idx]
        for token_idx in range(-1, -10, -1):
            
            X  = activations[layer_id][:,token_idx].numpy()
            Xn = X / np.sqrt((X ** 2).sum(axis=-1, keepdims=True))
            
            Xn_train, Xn_valid, y_train, y_valid = Xn[idx_train], Xn[idx_valid], y[idx_train], y[idx_valid]
            
            clf = LinearSVC(max_iter=10000).fit(Xn_train, y_train)
            acc  = metrics.roc_auc_score(y_valid, clf.decision_function(Xn_valid))
            out.append({"token_idx": token_idx, "layer_idx": layer_idx, "acc": acc, "it": it})

df_out = pd.DataFrame(out)
for it in df_out.it.unique():
    sub = df_out[df_out['it'] == it]
    _   = plt.plot(sub['token_idx'], sub['acc'], label=f'it {it}', alpha=0.25, c='grey')

_ = plt.plot(df_out.groupby('token_idx').acc.mean(), label='mean', c='black')

_ = plt.legend()
_ = plt.grid('both', c='grey', alpha=0.5)
_ = plt.xlabel('token index')
show_plot()

# Could you do this w/ BOW

# --

# One method:
#  1) Generate a bunch of random facts
#  2) Truncate at -n tokens
#  3) Add <choice>
#  4) Look at probability of rock / paper / scissors