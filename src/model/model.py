from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..util import arg_type


class LanguageModel(nn.Module):
    """
    abstract class for LSTM based captioning model
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.vocab = kwargs['vocab']

    def sample(self, **kwargs):
        input_feature = kwargs['input_feature']
        max_length = kwargs['max_length']
        sample_max = kwargs['sample_max']   # True or False

        start_word_id, end_word_id = self.vocab.start_token_id, self.vocab.end_token_id

        batch_size, input_feature = self.sample_prepare_feat(input_feature)

        last_state = self.sample_init_state(input_feature)
        log_prob_seq = []

        last_word_id_batch = [start_word_id] * batch_size
        word_id_seq = []

        unfinished_flag = [1 for _ in range(batch_size)]

        for t in range(max_length):
            output, state = self.step(input_feature=input_feature, last_word_id_batch=last_word_id_batch,
                                      last_state=last_state)
            log_prob = F.log_softmax(output, -1)    # (batch_size, vocab_size)

            if sample_max:
                word_log_prob, word_id = torch.max(log_prob, dim=-1)    # word_id: (64,)
            else:
                word_id = torch.multinominal(torch.exp(log_prob), num_samples=1)    # word_id: (batch_size, 1)
                word_log_prob = log_prob.gather(dim=1, index=word_id)               # word_log_prob: (batch_size, 1)
                word_id = word_id.squeeze(1)                    # word_id: (64,)
                word_log_prob = word_log_prob.squeeze(1)        # word_log_prob: (64,)

            if t == 0:
                unfinished_flag = word_id != end_word_id
            else:
                unfinished_flag = unfinished_flag * (word_id != end_word_id)

            _word_id = word_id.clone()
            _word_id[unfinished_flag == 0] = end_word_id

            word_id_seq.append(_word_id)
            last_word_id_batch = _word_id
            log_prob_seq.append(word_log_prob)

            if unfinished_flag.sum() == 0:
                break

        return word_id_seq, log_prob_seq

    def sample_beam(self, **kwargs):
        input_feature = kwargs['input_feature']
        max_length = kwargs['max_length']
        beam_size = kwargs['beam_size']

        start_word_id, end_word_id = self.vocab.start_token_id, self.vocab.end_token_id

        batch_size, input_feature = self.sample_prepare_feat(input_feature)
        assert(batch_size == 1)

        initial_state = self.sample_init_state(input_feature)
        # state, log_prob_sum, log_prob_seq, last_word_id, word_id_seq
        candidates = [(initial_state, 0., [], start_word_id, [])]

        for t in range(max_length):
            tmp_candidates = []
            end_flag = True
            for candidate in candidates:
                state, log_prob_sum, log_prob_seq, last_word_id, word_id_seq = candidate
                if last_word_id == end_word_id and t > 0:
                    tmp_candidates.append(candidate)
                else:
                    end_flag = False
                    output, state = self.step(input_feature=input_feature,
                                              last_word_id_batch=[last_word_id], last_state=state)

                    output = F.log_softmax(output, -1).squeeze(0).detach().cpu()  # log of probability
                    output_sorted, index_sorted = torch.sort(output, descending=True)
                    for k in range(beam_size):
                        log_prob, word_id = output_sorted[k], index_sorted[k]  # tensor, tensor
                        word_id = int(word_id.numpy())
                        tmp_candidates.append((state,
                                               log_prob_sum + log_prob, log_prob_seq + [log_prob],
                                               word_id, word_id_seq + [word_id]))
            candidates = sorted(tmp_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            # candidates = sorted(tmp_candidates, key=lambda x: x[1] / len(x[-1]), reverse=True)[:beam_size]
            if end_flag:
                break

        # log_prob_seq, word_id_seq
        return candidates[0][1], candidates[0][2], candidates[0][-1]

    @abstractmethod
    def sample_prepare_feat(self, input_feature):
        """
        :param input_feature: original_feature
        :return: batch_size, prepared_feature
        """
        return 0, None

    @abstractmethod
    def sample_init_state(self, input_feature):
        """
        perform steps before feeding <start> token
        :param input_feature:
        :return: initial lstm state
        """
        pass

    @abstractmethod
    @arg_type(last_word_id_batch=[np.array, list, tuple])
    def step(self, input_feature, last_word_id_batch, last_state):
        """
        :param last_word_id_batch: np.array | list | tuple
        :param last_state:
        :return: output (not softmax prob), state
        """
        return None, None
