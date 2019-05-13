from abc import abstractmethod
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..util import arg_type


BeamCandidate = namedtuple('BeamCandidate',
                           ['state', 'log_prob_sum', 'log_prob_seq', 'last_word_id', 'word_id_seq', 'metadata_seq'])


class LanguageModel(nn.Module):
    """
    abstract class for LSTM based captioning model
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.vocab = kwargs['vocab']

    @arg_type(input_sentence=torch.Tensor)
    def forward(self, input_feature, input_sentence, **kwargs):
        batch_size, input_feature = self.prepare_feat(input_feature, **kwargs)
        assert len(input_sentence.shape) == 2 and input_sentence.shape[0] == batch_size
        max_length = input_sentence.shape[1]

        state = self.init_state(input_feature, **kwargs)
        all_outputs = []
        for i in range(max_length):
            word_id_batch = input_sentence[:, i]
            output, state, step_metadata = self.step(input_feature, word_id_batch, state, **kwargs)    # output: (batch_size, vocab_size)
            all_outputs.append(output)

        all_outputs = torch.stack([all_outputs], 1)     # (batch_size, max_len, vocab_size)
        return all_outputs

    def sample(self, input_feature, max_length, sample_max, **kwargs):
        """

        :param input_feature:
        :param max_length:
        :param sample_max:
        :param kwargs:
        :return: log_prob_seq (np.array), word_id_seq (np.array), all_metadata (list)
        """

        start_word_id, end_word_id = self.vocab.start_token_id, self.vocab.end_token_id

        batch_size, input_feature = self.prepare_feat(input_feature, **kwargs)

        last_state = self.init_state(input_feature, **kwargs)
        log_prob_seq = []
        log_prob_sum = []

        last_word_id_batch = [start_word_id] * batch_size
        word_id_seq = []

        unfinished_flag = [1 for _ in range(batch_size)]

        all_metadata = []

        for t in range(max_length):
            output, state, step_metadata = self.step(input_feature=input_feature,
                                                     last_word_id_batch=last_word_id_batch,
                                                     last_state=last_state, **kwargs)
            log_prob = F.log_softmax(output, -1)    # (batch_size, vocab_size)

            if sample_max:
                word_log_prob, word_id = torch.max(log_prob, dim=-1)    # word_id: (batch_size,)
            else:
                word_id = torch.multinominal(torch.exp(log_prob), num_samples=1)    # word_id: (batch_size, 1)
                word_log_prob = log_prob.gather(dim=1, index=word_id)               # word_log_prob: (batch_size, 1)
                word_id = word_id.squeeze(1)                    # word_id: (batch_size,)
                word_log_prob = word_log_prob.squeeze(1)        # word_log_prob: (batch_size,)

            if t == 0:
                unfinished_flag = word_id != end_word_id
            else:
                unfinished_flag = unfinished_flag * (word_id != end_word_id)

            _word_id = word_id.clone()
            _word_id[unfinished_flag == 0] = end_word_id

            word_id_seq.append(_word_id)
            last_word_id_batch = _word_id
            log_prob_seq.append(word_log_prob)

            all_metadata.append(step_metadata)

            if unfinished_flag.sum() == 0:
                break

        log_prob_seq = torch.stack(log_prob_seq, dim=1)     # (batch_size, seq_len)
        word_id_seq = torch.stack(word_id_seq, dim=1)       # (batch_size, seq_len)

        log_prob_seq = log_prob_seq.detach().cpu().numpy()
        word_id_seq = word_id_seq.detach().cpu().numpy()

        return log_prob_seq, word_id_seq, all_metadata

    def sample_beam(self, input_feature, max_length, beam_size, **kwargs):
        """

        :param input_feature:
        :param max_length:
        :param beam_size:
        :param kwargs:
        :return: log_prob_seq (np.array), word_id_seq (np.array), all_metadata (list)
        """
        start_word_id, end_word_id = self.vocab.start_token_id, self.vocab.end_token_id

        batch_size, input_feature = self.prepare_feat(input_feature, **kwargs)
        assert(batch_size == 1)

        initial_state = self.init_state(input_feature, **kwargs)
        # state, log_prob_sum, log_prob_seq, last_word_id, word_id_seq, metadata_seq
        candidates = [BeamCandidate(initial_state, 0., [], start_word_id, [], [])]

        for t in range(max_length):
            tmp_candidates = []
            end_flag = True
            for candidate in candidates:
                state, log_prob_sum, log_prob_seq, last_word_id, word_id_seq, step_metadata_history = candidate
                if last_word_id == end_word_id and t > 0:
                    tmp_candidates.append(candidate)
                else:
                    end_flag = False
                    output, state, step_metadata = self.step(input_feature=input_feature,
                                                             last_word_id_batch=[last_word_id],
                                                             last_state=state, **kwargs)

                    output = F.log_softmax(output, -1).squeeze(0).detach().cpu()  # log of probability
                    output_sorted, index_sorted = torch.sort(output, descending=True)
                    for k in range(beam_size):
                        log_prob, word_id = output_sorted[k], index_sorted[k]  # tensor, tensor
                        word_id = int(word_id.numpy())
                        log_prob = float(log_prob.numpy())
                        tmp_candidates.append(BeamCandidate(state,
                                               log_prob_sum + log_prob, log_prob_seq + [log_prob],
                                               word_id, word_id_seq + [word_id],
                                               step_metadata_history + [step_metadata]))
            candidates = sorted(tmp_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            # candidates = sorted(tmp_candidates, key=lambda x: x[1] / len(x[-1]), reverse=True)[:beam_size]
            if end_flag:
                break

        # log_prob_seq, word_id_seq, metadata_seq
        return np.array(candidates[0].log_prob_seq), np.array(candidates[0].word_id_seq), candidates[0].metadata_seq

    @abstractmethod
    def prepare_feat(self, input_feature, **kwargs):
        """
        prepare input_feature for next steps
        :param input_feature: original_feature
        :return: batch_size, prepared_feature
        """
        return 0, None

    @abstractmethod
    def init_state(self, input_feature, **kwargs):
        """
        perform steps before feeding <start> token
        :param input_feature:
        :return: initial lstm state
        """
        pass

    @abstractmethod
    @arg_type(last_word_id_batch=[np.array, list, tuple])
    def step(self, input_feature, last_word_id_batch, last_state, **kwargs):
        """
        :param input_feature: returned by sample_prepare_feat, batched
        :param last_word_id_batch: batched
        :param last_state: batched
        :return: output (without softmax, shape is [batch_size, vocab_size]),
                 state,
                 step_metadata
        """
        return None, None, None
