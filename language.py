from matplotlib import pyplot as plt
from rcode import *

import json
import torch
import transformer_lens
from rich import print as rprint

torch.set_grad_enabled(False)
from transformers import GenerationConfig

import numpy as np
from seaborn import heatmap
from torch.nn.functional import normalize

# --

padding_side = 'left'

model_str                    = "Qwen/Qwen3-8B"
model                        = transformer_lens.HookedTransformer.from_pretrained(model_str)
model.tokenizer.padding_side = padding_side
model.tokenizer.pad_token    = model.tokenizer.eos_token

def n_tokens(x):
    return len(model.tokenizer.encode(x))

# --
# Generation

model.reset_hooks()

traces = [
    # English facts
    "Elephants can weigh up to 6 tons and live for 70 years.",                    # 0
    "Dolphins use echolocation to navigate and hunt for food.",                   # 1  
    "Hummingbirds can fly backwards and beat their wings 80 times per second.",   # 2
    "Octopuses have three hearts and blue blood.",                               # 3
    "Penguins can dive up to 500 meters deep to catch fish.",                   # 4
    
    # German translations  
    "Elefanten können bis zu 6 Tonnen wiegen und 70 Jahre alt werden.",         # 5
    "Delfine nutzen Echoortung um zu navigieren und Nahrung zu jagen.",          # 6
    "Kolibris können rückwärts fliegen und ihre Flügel 80 Mal pro Sekunde schlagen.", # 7
    "Oktopusse haben drei Herzen und blaues Blut.",                             # 8
    "Pinguine können bis zu 500 Meter tief tauchen um Fische zu fangen."        # 9
]
messages = [
    [
        {"role": "user", "content": trace}
    ]
    for trace in traces
]

input_str = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

logits, activations = model.run_with_cache(input_str, padding_side=padding_side, names_filter=lambda hook_name: 'resid_post' in hook_name)
activations         = activations.to('cpu')
logits              = logits.to('cpu')


# min_tokens = min([n_tokens(s) for s in input_str])

import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import math

# cosine distance between average activations
layer_keys = [k for k in activations.keys() if 'resid_post' in k]
layer_keys = layer_keys[20:]
num_layers = len(layer_keys)
grid_size = math.ceil(math.sqrt(num_layers))

plt.figure(figsize=(20, 20))
for idx, k in enumerate(layer_keys):
    plt.subplot(grid_size, grid_size, idx + 1)
    
    a = activations[k]
    # a = a[:, :min_tokens]
    a = a.numpy()
    
    d = squareform(pdist(a[:, -4:].mean(axis=1), metric='cosine'))
    d = d / d.mean()
    heatmap(d, cmap='viridis')
    plt.title(f"Layer: {k}")

plt.tight_layout()
show_plot()


# cosine distance between last activations
layer_keys = [k for k in activations.keys() if 'resid_post' in k]
layer_keys = layer_keys[20:]
num_layers = len(layer_keys)
grid_size = math.ceil(math.sqrt(num_layers))

plt.figure(figsize=(20, 20))
for idx, k in enumerate(layer_keys):
    plt.subplot(grid_size, grid_size, idx + 1)
    
    a = activations[k]
    # a = a[:, :min_tokens]
    a = a.numpy()
    
    d = squareform(pdist(a[:, -1], metric='cosine'))
    d = d / d.mean()
    heatmap(d, cmap='viridis')
    plt.title(f"Layer: {k}")

plt.tight_layout()
show_plot()
