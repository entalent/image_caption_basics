import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .model import LanguageModel

class LSTMLanguageModel(LanguageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_embedding = nn.Linear(in_features=2048, out_features=300)
        self.input_embedding = nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=300, padding_idx=0)
        self.lstm = nn.LSTMCell(input_size=300, hidden_size=512)
        self.output_embedding = nn.Linear(in_features=512, out_features=len(self.vocab))

    def prepare_feat(self, input_feature, **kwargs):
        batch_size = len(input_feature)
        prepared_feat = self.image_embedding(input_feature)
        return batch_size, prepared_feat

    def init_state(self, input_feature, **kwargs):
        return self.lstm(input_feature)

    def step(self, input_feature, last_word_id_batch, last_state, **kwargs):
        device = input_feature.device
        last_word_id_batch = torch.LongTensor(np.array(last_word_id_batch).astype(np.int64)).to(device)
        emb = self.input_embedding(last_word_id_batch)
        output, state = self.lstm(emb, last_state)
        output = self.output_embedding(output)
        return output, state, None


