# %% [markdown]
# ## Encoder and Self-Attention
# %%
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel
)

from bertviz import head_view
from bertviz.transformers_neuron_view import BertModel
from bertviz.neuron_view import show

import torch
from torch import nn
from torch.nn import functional as F

from math import sqrt

# %%
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BertModel.from_pretrained(model_ckpt)
text = "time flies like an arrow"

show(
    model,
    "bert",
    tokenizer,
    text,
    display_mode="light",
    layer=0,
    head=8
)

# %% [markdown]
# ## Implementation of Transformer architecture
# %%
# Tokenized input
text = "time flies like an arrow"
inputs = tokenizer(
    text,
    return_tensors="pt",
    add_special_tokens=False
)
inputs.input_ids

# %%
# Tokenized input to the same word give us the same result all the time
text = "time time time time time"
inputs2 = tokenizer(
    text,
    return_tensors="pt",
    add_special_tokens=False
)
inputs2.input_ids

# %%
config = AutoConfig.from_pretrained(model_ckpt)
config.vocab_size, config.hidden_size

# %%
# Create config and Embedding layer for tokens
config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
token_emb

# %%
input_emb = token_emb(inputs.input_ids)
input_emb.shape

# %%
input_emb

# %%
query = key = value = input_emb
dim_k = key.size(-1)
scores = torch.bmm(query, key.transpose(1,2)) / sqrt(dim_k)
scores.size()

# %%
scores

# %%
weights = F.softmax(scores, dim=-1)
weights.sum(-1)

# %%
weights.sum(-1, keepdim=True)

# %%
attn_outputs = torch.bmm(weights, value)
attn_outputs.shape

# %%
# Let's wrap these steps into a function that we can use later
def scaled_dot_product_attention(query, key, value):
    dim_k = key.size(-1)
    scores = torch.bmm(query, key.transpose(1,2)) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)


# %% [markdown]
# ### Multi-Head attention
# %%
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)
    
    def forward(self, hidden_states):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_states),
            self.k(hidden_states),
            self.v(hidden_states)
        )
        return attn_outputs    

# %%
attention_head = AttentionHead(
    config.hidden_size,
    config.hidden_size // config.num_attention_heads
)
attention_head_outputs = attention_head(input_emb)
attention_head_outputs.shape


# %%
class MultiAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([
            AttentionHead(embed_dim, head_dim)
            for _ in range(num_heads)
        ])
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_states):
        x = torch.cat([h(hidden_states) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x


# %%
multihead_attn = MultiAttentionHead(config)
attn_outputs = multihead_attn(input_emb)
attn_outputs.shape


# %%
model = AutoModel.from_pretrained(
    model_ckpt, output_attentions=True
)

sentence_a = "time flies like an arrow"
sentence_b = "fruit flies like a banana"

viz_inputs = tokenizer(
    sentence_a,
    sentence_b,
    return_tensors='pt'
)
attention = model(**viz_inputs).attentions

sentence_b_start = (viz_inputs.token_type_ids == 0).sum(dim=1)

tokens =  tokenizer.convert_ids_to_tokens(viz_inputs.input_ids[0])

head_view(attention, tokens, sentence_b_start, heads=[8])


# %% [markdown]
# ### The Feed-Forward Layer
# %%
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.hidden_size,
            config.intermediate_size
        )
        self.linear_2 = nn.Linear(
            config.intermediate_size,
            config.hidden_size
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(
            config.hidden_dropout_prob
        )

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
    

# %%
feed_forward = FeedForward(config)
ff_outputs = feed_forward(attn_outputs)
ff_outputs.size()

# %% [markdown]
# ### Layer Normalization
# %%
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.multihead_attn = MultiAttentionHead(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        hidden_states = self.layer_norm_1(x)
        x = x + self.multihead_attn(hidden_states)
        x = x + self.feed_forward(
            self.layer_norm_2(x)
        )
        return x


# %%
encoder_layer = TransformerEncoderLayer(config)
input_emb.shape, encoder_layer(input_emb).size()

# %% [markdown]
# ### Positional embedding
# %%
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(
            config.vocab_size,
            config.hidden_size
        )
        self.positional_emb = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        self.layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=1e-12
        )
        self.dropout = nn.Dropout()

    def forward(self, x):
        # Create position IDs for input sequence
        seq_len = x.size(1)
        pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        # Create token and position embeddings
        token_emb = self.token_emb(x)
        pos_emb = self.positional_emb(pos_ids)
        # Combine token and position embeddings
        embs = self.layer_norm(token_emb + pos_emb)
        embs = self.dropout(embs)
        return embs

# %%
embeddings_layer = Embeddings(config)
embeddings_layer(inputs.input_ids).shape

# %%
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x


# %%
encoder = TransformerEncoder(config)
encoder(inputs.input_ids).shape

# %% [markdown]
# ### Adding a Classification Head
# %%
class TransformerForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.encoder(x)[:, 0, :]
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# %%
config.num_labels = 3
encoder_classifier = TransformerForSequenceClassification(config)
encoder_classifier(inputs.input_ids).size()

# %% [markdown]
# ### The Decoder
# %%
seq_len = inputs.input_ids.size(-1)
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
mask[0]

# %%
scores.masked_fill(mask == 0, -float("inf"))

# %%
def scaled_dot_product_attention(query, key, value, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights.bmm(value)


# %%

# %%
