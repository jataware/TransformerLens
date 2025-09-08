import numpy as np
from rich import print as rprint

import transformer_lens

class InspectModel:
    def __init__(self, model_str='Qwen/Qwen3-8B', padding_side='right'):
        self.model_str                    = model_str
        self.model                        = transformer_lens.HookedTransformer.from_pretrained(model_str)
        self.model.tokenizer.pad_token    = self.model.tokenizer.eos_token
        
        self.model.tokenizer.padding_side = padding_side
    
    def n_tokens(self, x):
        if isinstance(x, list):
            return [self.n_tokens(xx) for xx in x]
        
        return len(self.model.tokenizer.encode(x))
    
    def prep(self, messages):
        return self.model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    def forward(self, messages):
        n_messages = len(messages)
        inputs_str = self.prep(messages)
        n_tokens   = self.n_tokens(inputs_str)
        
        self.model.reset_hooks()
        logits, activations = self.model.run_with_cache(
            inputs_str, 
            padding_side        = self.model.tokenizer.padding_side, 
            names_filter        = lambda hook_name: 'resid' in hook_name,
            return_cache_object = False
        )
        
        _cache = {
            **{k:v.to('cpu') for k, v in activations.items()},
            "logits": logits.to('cpu'),
        }
        
        out = []
        for i in range(n_messages):
            if self.model.tokenizer.padding_side == 'right':
                tmp = {k: _cache[k][i,:n_tokens[i]] for k in _cache.keys()}
            elif self.model.tokenizer.padding_side == 'left':
                tmp = {k: _cache[k][i,-n_tokens[i]:] for k in _cache.keys()}
            
            out.append(tmp)
        
        return out
    
    def batched_forward(self, messages, tokens_per_batch=1024):
        inputs_str = self.prep(messages)
        n_tokens   = self.n_tokens(inputs_str)
        assert max(n_tokens) <= tokens_per_batch, "Max token count per batch is less than the max token count per message"
        
        # Sort by token count for efficient batching
        asort           = np.argsort(n_tokens)
        sorted_messages = [messages[i] for i in asort]
        sorted_n_tokens = [n_tokens[i] for i in asort]
        
        # --
        # Create batches
        print('batched_forward: creating batches')
        
        batches     = []
        curr_batch  = []
        curr_n_toks = 0
        for i, (msg, tokens) in enumerate(zip(sorted_messages, sorted_n_tokens)):
            if curr_n_toks + tokens > tokens_per_batch and curr_batch:
                batches.append(curr_batch)
                print(f'\tbatch={curr_batch} | n_tokens={curr_n_toks}')
                curr_batch  = []
                curr_n_toks = 0
            
            curr_batch.append(i)
            curr_n_toks += tokens
        
        if curr_batch:
            batches.append(curr_batch)
            print(f'\tbatch={curr_batch} | n_tokens={curr_n_toks}')
            
        # --
        # Run
        
        all_results = [None] * len(messages)
        for batch_indices in batches:
            batch_messages = [sorted_messages[i] for i in batch_indices]
            batch_results  = self.forward(batch_messages)
            
            for batch_idx, result in zip(batch_indices, batch_results):
                all_results[asort[batch_idx]] = result
        
        return all_results
        


if __name__ == '__main__':
    ins = InspectModel(model_str='Qwen/Qwen3-0.6B')
    messages = [
        [
            {'role': 'user', 'content': 'What is the capital of France?'}
        ],
        [
            {'role': 'user', 'content': 'What is the capital of the most populous state of Germany?'}
        ],
        [
            {'role': 'user', 'content': 'What is the capital of France?'}
        ],
        [
            {'role': 'user', 'content': 'What is the capital of the most populous state of Germany?'}
        ],
        [
            {'role': 'user', 'content': 'What is the capital of France?'}
        ],
        [
            {'role': 'user', 'content': 'What is the capital of the most populous state of Germany?'}
        ],        
    ]
    cache = ins.batched_forward(messages, tokens_per_batch=51)
    