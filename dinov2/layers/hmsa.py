# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn


# Define the single prompted attention layer
class PromptedAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, prompt_dim):
        super(PromptedAttentionLayer, self).__init__()
        # MultiheadAttention Layer from PyTorch
        self.attn = nn.MultiheadAttention(d_model , num_heads)


    def forward(self, x, y, prompt):
        # Modify queries with the prompt
        #print (x.shape, y.shape, prompt.shape)
        prompt_expanded = prompt.unsqueeze(0).expand(x.shape[0], -1, -1)  # Match sequence length
        queries = torch.cat([x, prompt_expanded], dim=-1)  # Concatenate along feature dimension
  
        
        # Compute attention (query, key, value)
        attn_output, _ = self.attn(queries, y, y)  # Use y as key and value
        
        # Residual connection (Add original input to the attention output)
        output = attn_output + x
        return output

# Define the multi-layer prompted attention model
class HMSA_head(nn.Module):
    def __init__(self, num_layers=4, d_model=1024, num_heads = 4, prompt_dim = 1024):
        super(HMSA_head, self).__init__()
        self.num_layers = num_layers
        # Stack multiple prompted attention layers
        self.attention_layers = nn.ModuleList([
            PromptedAttentionLayer(d_model, num_heads, prompt_dim) for _ in range(num_layers)
        ])

    def forward(self, x, y, prompt):
        output = x
        # Pass through each layer of prompted attention
        for layer in self.attention_layers:
            output = layer(output, y, prompt)
        return output