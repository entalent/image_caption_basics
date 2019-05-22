import os
import sys
import pickle

import numpy as np

glove_embedding_file = r'/media/wentian/sdb2/work/caption_dataset/glove/glove.6B.300d.txt'

all_embedding = {}
with open(glove_embedding_file, 'r') as f:
    for line in f:
        if len(line) == 0:
            continue
        strs = line.strip().split()
        assert (len(strs) == 301)
        word = strs[0]
        vector = np.array([float(i) for i in strs[1:]]).astype(np.float32)

        all_embedding[word] = vector

with open('../data/glove.6B.300d.preprocessed.pkl', 'wb') as f:
    pickle.dump(all_embedding, f)