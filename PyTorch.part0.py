#!/usr/bin/env python
# coding: utf-8

# In[1]:


# GPT5 : extracting weights and biases from each layer in PyTorch is straightforward, 
# and you can do it in several ways depending on how you structured your model.


# In[2]:


print('''
| Method             | Code                                          | Best for         |
| ------------------ | --------------------------------------------- | ---------------- |
| Direct             | `model.fc1.weight`                            | Few known layers |
| Loop               | `for name, param in model.named_parameters()` | Medium models    |
| `state_dict()`     | `model.state_dict()`                          | Saving / loading |
| List comprehension | `[p for n, p in model.named_parameters()]`    | Quick extraction |
''')


# In[3]:


print('''

You can get every layer’s weights and biases through .named_parameters() or .state_dict() 
— both return everything as tensors that you can inspect, modify, or save.

''')


# In[4]:


import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet()


# In[5]:


# Access weights and biases layer by layer


# In[6]:


# Option A — Direct attribute access

print("Layer 1 weight:\n", model.fc1.weight)
print("Layer 1 bias:\n", model.fc1.bias)

print("\nLayer 2 weight:\n", model.fc2.weight)
print("Layer 2 bias:\n", model.fc2.bias)


# In[7]:


# You can also convert them to NumPy arrays:

w1 = model.fc1.weight.detach().numpy()
b1 = model.fc1.bias.detach().numpy()

print(w1)
print(b1)


# In[8]:


# You can also convert them to NumPy arrays:

w2 = model.fc2.weight.detach().numpy()
b2 = model.fc2.bias.detach().numpy()

print(w2)
print(b2)


# In[9]:


print(model.named_parameters())


# In[10]:


# Option B — Using a loop over all layers

for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"{name} → weight:\n{param.data}")
    elif 'bias' in name:
        print(f"{name} → bias:\n{param.data}")


# In[11]:


# Option C — Accessing via .state_dict()

# state_dict() is a dictionary containing all model parameters (weights & biases):

for name, tensor in model.state_dict().items():
    print(name, ":", tensor.shape)


# In[12]:


state_dict = model.state_dict()
print(state_dict)


# In[14]:


# Option D — Store weights/biases in lists


# In[13]:


weights = [p.data for n, p in model.named_parameters() if 'weight' in n]
biases  = [p.data for n, p in model.named_parameters() if 'bias' in n]

print("All weights:", weights)
print("All biases:", biases)


# In[14]:


# For convolutional or transformer layers

for name, param in model.named_parameters():
    print(name, param.shape)


# In[15]:


def list_params(model):
    for name, p in model.named_parameters():
        print(f"{name:60s} {tuple(p.shape)}  requires_grad={p.requires_grad}")

list_params(model)  # your nn.Transformer / custom model


# In[16]:


print('''

To extract a Transformer’s weights, biases, and attention parameters (and optionally the attention maps) in PyTorch.

''')


# In[17]:


from transformers import AutoModel, AutoTokenizer
import torch

name = "bert-base-uncased"  
tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name, output_attentions = True)
model.eval()

inputs = tok("Hello world!", return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)

# Attention maps: list over layers
# Each item: (batch, num_heads, seq_len, seq_len)
attn_maps = out.attentions
print(len(attn_maps), attn_maps[0].shape)


# In[25]:


# Inspect / extract attention parameters per layer

# BERT naming example:
# encoder.layer.{L}.attention.self.query.weight / .bias
# ...key.weight, ...value.weight
# ...output.dense.weight / .bias


# In[18]:


for name, p in model.named_parameters():
    if any(k in name for k in ["query", "key", "value", "out_proj", "output.dense"]):
        print(name, tuple(p.shape))


# In[19]:


# Example: get the Q/K/V and output projection for layer L:


# In[20]:


L = 0
qry_W = dict(model.named_parameters())[f"encoder.layer.{L}.attention.self.query.weight"]
qry_b = dict(model.named_parameters())[f"encoder.layer.{L}.attention.self.query.bias"]
key_W = dict(model.named_parameters())[f"encoder.layer.{L}.attention.self.key.weight"]
val_W = dict(model.named_parameters())[f"encoder.layer.{L}.attention.self.value.weight"]
out_W = dict(model.named_parameters())[f"encoder.layer.{L}.attention.output.dense.weight"]
out_b = dict(model.named_parameters())[f"encoder.layer.{L}.attention.output.dense.bias"]


# In[21]:


print("=== Layer", L, "===")
print("Query weight:", qry_W.shape)
print("Query bias:", qry_b.shape)
print("Key weight:", key_W.shape)
print("Value weight:", val_W.shape)
print("Output weight:", out_W.shape)
print("Output bias:", out_b.shape)


# In[22]:


import pandas as pd

pd.DataFrame({
    "Param": ["query.weight", "query.bias", "key.weight", "value.weight", "output.weight", "output.bias"],
    "Shape": [tuple(qry_W.shape), tuple(qry_b.shape), tuple(key_W.shape), tuple(val_W.shape), tuple(out_W.shape), tuple(out_b.shape)]
})


# In[23]:


# If you want to print all attention weights/biases for every layer, just loop:

# for L in range(model.config.num_hidden_layers):
#    print(f"\n=== Layer {L} ===")
#    layer_params = dict(model.named_parameters())
#    for name in ["query", "key", "value"]:
#        w = layer_params[f"encoder.layer.{L}.attention.self.{name}.weight"]
#        b = layer_params[f"encoder.layer.{L}.attention.self.{name}.bias"]
#        print(f"{name}: {tuple(w.shape)} / {tuple(b.shape)}")


# In[24]:


# !pip install transformers torch matplotlib

import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"  # or "gpt2", etc.
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True).eval()

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")

with torch.no_grad():
    out = model(**inputs)

# Choose a layer and head
L = 0             # layer index (0-based)
head = 0          # head index (0-based)

# Shape: list over layers; each tensor: (batch, heads, seq_len, seq_len)
att = out.attentions[L][0, head]              # (seq_len, seq_len)
# Or average across heads:
# att = out.attentions[L][0].mean(dim=0)      # (seq_len, seq_len)

# Token labels for axes
tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0])

# Plot (matplotlib; one figure, no specific colors)
plt.figure(figsize=(6, 5))
plt.imshow(att.cpu(), aspect='auto')
plt.title(f"Attention heatmap — layer {L}, head {head}")
plt.xticks(range(len(tokens)), tokens, rotation=90)
plt.yticks(range(len(tokens)), tokens)
plt.xlabel("Key / Source positions")
plt.ylabel("Query / Target positions")
plt.tight_layout()
plt.show()


# In[ ]:




