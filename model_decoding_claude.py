import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GemmaForCausalLM, GemmaTokenizer
import math
import numpy as np

""" SOTA EEG-To-Text Decoding Architecture upgrading from BART to Gemma 2 (Causal Decoder-Only).

    Key architectural adaptation for Causal LLMs (Gemma 2):
    - Unlike BART (Encoder-Decoder), Gemma is a Causal Decoder-Only LLM.
    - During training, continuous EEG prefix embeddings are concatenated with the target token embeddings
      so the causal model learns to autoregressively generate text conditioned on the EEG brain signals.
    - During generation (`generate`), decoding starts from the projected EEG prefix embeddings.
"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class BrainTranslatorGemma(nn.Module):
    def __init__(self, pretrained_layers, in_feature=840, decoder_embedding_size=2304, additional_encoder_nhead=8,
                 additional_encoder_dim_feedforward=2048):
        super(BrainTranslatorGemma, self).__init__()

        self.pretrained = pretrained_layers # Gemma2ForCausalLM

        # EEG Encoder
        self.positional_embedding = PositionalEncoding(in_feature)
        self.additional_encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,
                                                                   dim_feedforward=additional_encoder_dim_feedforward,
                                                                   batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)

        # Projection layer: maps EEG encoder output to Gemma 2 hidden size (2304 for Gemma-2-9b)
        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def addin_forward(self, input_embeddings_batch, input_masks_invert):
        # EEG feature extraction with positional encoding
        x = self.positional_embedding(input_embeddings_batch)
        encoded_embedding = self.additional_encoder(x, src_key_padding_mask=input_masks_invert)

        # Projection to Gemma latent space
