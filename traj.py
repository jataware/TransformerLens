from matplotlib import pyplot as plt
from rcode import *

import json
import torch
import pandas as pd
import transformer_lens
from rich import print as rprint

torch.set_grad_enabled(False)
from transformers import GenerationConfig

import numpy as np
from seaborn import heatmap
from torch.nn.functional import normalize

# --

padding_side = 'right'

model_str                    = "microsoft/phi-4"
model                        = transformer_lens.HookedTransformer.from_pretrained(model_str)
model.tokenizer.padding_side = padding_side
model.tokenizer.pad_token    = model.tokenizer.eos_token

def n_tokens(x):
    return len(model.tokenizer.encode(x))

# --
# Generation

model.reset_hooks()

traces = [json.loads(line) for line in open('trace2.jl')]
messages = [
    [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": trace['prompt']},
        {"role": "assistant", "content": trace['output_str']}
    ]
    for trace in traces
]

input_str = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
input_str = input_str[-4:] # only last three examples

logits, activations = model.run_with_cache(input_str, padding_side=padding_side, names_filter=lambda hook_name: 'resid_post' in hook_name)
activations         = activations.to('cpu')
logits              = logits.to('cpu')

# for k in activations.keys():
#     print(k, activations[k].shape)

min_tokens = min([n_tokens(s) for s in input_str])
pre_tokens = n_tokens(model.tokenizer.apply_chat_template(messages[0][:2], tokenize=False, add_generation_prompt=False))


a = activations['blocks.30.hook_resid_post'].clone()
a = a[:, pre_tokens:min_tokens]

f = a[:, -1].mean(axis=1)
d = squareform(pdist(f, metric='cosine'))
d.round(2) * 100


a  = a.numpy()
an = a
# an = np.array([pd.DataFrame(x).rolling(window=32).mean().values for x in a])
an = an / np.sqrt((an ** 2).sum(axis=-1, keepdims=True))

labs = [0, 1, 1, 0]
for i in range(an.shape[0]):
    for j in range(i + 1, an.shape[0]):
        c = 'red' if labs[i] == labs[j] else 'black'
        z = 1 - (an[i] * an[j]).sum(axis=-1)
        z = pd.Series(z).rolling(window=10).mean().values
        _ = plt.plot(z, label=f'{i} vs {j}', c=c)

_ = plt.legend()
show_plot()


