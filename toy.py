from rich import print as rprint

import torch
import transformer_lens
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES

torch.set_grad_enabled(False)

# rprint(OFFICIAL_MODEL_NAMES)

# --



# --

model = transformer_lens.HookedTransformer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")

input_str = "My name is"
logits, activations = model.run_with_cache(input_str)

logits = logits.squeeze()

model.tokenizer.convert_ids_to_tokens(logits[3].argsort(axis=-1)[-32:])
