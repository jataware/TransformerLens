import transformer_lens

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
        logits      = logits.to('cpu')
        activations = activations.to('cpu')
        
        return logits, activations
    
    def batched_forward(self, batch_messages, batch_size=None):
        if batch_size is None:
            batch_size = len(batch_messages)
        
        all_logits = []
        all_activations = []
        
        for i in range(0, len(batch_messages), batch_size):
            batch = batch_messages[i:i+batch_size]
            prepared_batch = [self.prep(messages) for messages in batch]
            
            self.model.reset_hooks()
            logits, activations = self.model.run_with_cache(
                prepared_batch,
                padding_side = self.model.tokenizer.padding_side, 
                names_filter = lambda hook_name: 'resid_post' in hook_name
            )
            
            logits = logits.to('cpu')
            activations = activations.to('cpu')
            
            all_logits.append(logits)
            all_activations.append(activations)
        
        return all_logits, all_activations

if __name__ == '__main__':
    ins = InspectModel(model_str='Qwen/Qwen3-0.6B')
    messages = [
        [
            {'role': 'user', 'content': 'What is the capital of France?'}
        ]
    ]
    logits, activations = ins.forward(messages)
    print(logits.shape)
    print(activations.shape)