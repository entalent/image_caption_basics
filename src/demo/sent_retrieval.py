import os
import random
import sys
from collections import defaultdict

sys.path.append('.')

import torch
import torch.nn as nn
import torch.utils.data.dataloader
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

import util
from util import arg_type
from util import BasePipeline
import numpy as np
from model.model import LanguageModel


from demo1 import CaptionDataset, collate_fn_group


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class SentenceEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        vocab = kwargs['vocab']
        vocab_size = len(vocab)
        word_embedding_dim = kwargs['word_embedding_dim']
        hidden_size = kwargs['hidden_size']
        self.word_embedding_dim = word_embedding_dim
        self.hidden_size = hidden_size

        self.input_embedding = kwargs['input_embedding']
        self.encoder = nn.GRUCell(input_size=word_embedding_dim, hidden_size=hidden_size)

    @arg_type(sent=torch.Tensor, sent_length=torch.Tensor)
    def forward(self, sent, sent_length):
        """
        :param sent: (batch_size, max_len)
        :param sent_length: (batch_size)
        :return:
        """
        batch_size, max_len = sent.shape
        device = sent.device

        state = torch.zeros(batch_size, self.hidden_size).to(device)

        all_output = []
        for i in range(max_len):
            input_word = sent[:, i]     # (batch_size,)
            input_emb = self.input_embedding(input_word)  # (batch_size, emb_dim)
            state = self.encoder(input_emb, state)  # state = (h, c) h: (batch_size, hidden_dim)

            all_output.append(state)

        all_output = torch.stack(all_output, dim=1)     # (batch_size, max_len, hidden_dim)
        index = (sent_length - 1).unsqueeze(1).unsqueeze(2)   # (batch_size, 1, 1)
        index = index.expand(batch_size, 1, self.hidden_size)    # (batch_size, 1, hidden_dim)
        encoder_output = all_output.gather(dim=1, index=index)  # (batch_size, 1, hidden_dim)
        encoder_output = encoder_output.squeeze(1)      # (batch_size, hidden_dim)

        encoder_output = l2norm(encoder_output)         # (batch_size, hidden_dim)
        return encoder_output


class SentenceDecoder(LanguageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        word_embedding_dim = kwargs['word_embedding_dim']
        hidden_size = kwargs['hidden_size']
        self.input_embedding = kwargs['input_embedding']    # shared with encoder
        self.word_embedding_dim = word_embedding_dim
        self.hidden_size = hidden_size

        self.decoder = nn.GRUCell(input_size=self.word_embedding_dim, hidden_size=hidden_size)
        self.output_embedding = nn.Linear(in_features=hidden_size, out_features=len(self.vocab))

    def prepare_feat(self, input_feature, **kwargs):
        batch_size = input_feature.shape[0]
        return batch_size, input_feature

    def init_state(self, input_feature, **kwargs):
        return input_feature

    def step(self, input_feature, last_word_id_batch, last_state, **kwargs):
        emb = self.input_embedding(last_word_id_batch)
        state = self.decoder(emb, last_state)
        output = self.output_embedding(state)

        return output, state, None


class SentenceMatcher(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        vocab = kwargs['vocab']
        word_embedding_dim = kwargs['word_embedding_dim']
        self.input_embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=word_embedding_dim)
        kwargs['input_embedding'] = self.input_embedding
        self.encoder = SentenceEncoder(**kwargs)
        self.decoder = SentenceDecoder(**kwargs)

    def forward(self, sent, sent_len):
        """
        :param sent: (batch_size, max_len), including <start> and <end>
        :param sent_len:
        :return:
        """

        encoded = self.get_sent_embedding(sent, sent_len)
        output = self.decoder(input_feature=encoded, input_sentence=sent[:, :-1])
        return output

    def get_sent_embedding(self, sent, sent_len):
        """
        :param sent: (batch_size, max_len), including <start> and <end>
        :param sent_len:
        :return:
        """
        encoded = self.encoder(sent, sent_len)
        return encoded


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# class RetrievalPipeline(BasePipeline):
#     def __init__(self):
#         super().__init__()
#
#     def add_arguments(self, parser):
#         super().add_arguments(parser)
#
#     def run(self):
#         super().run()
#         self.train()
#
#     def train(self):
#         vocab = util.load_custom(r'../data/vocab/vocab_merged_coco_cub200.json')
#
#         train_dataset = CaptionDataset(dataset_name='coco', vocab=vocab, split='train',
#                                        image_mode='none', iter_mode='single', max_sent_length=18)
#         train_dataloader = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=512, shuffle=True,
#                                                                   collate_fn=util.custom_collate_fn)
#
#         model = SentenceMatcher(use_same_encoder=True, vocab=vocab, margin=0.2, measure='cosine', max_violation=True)
#         model.to(device)
#
#         optimizer = torch.optim.Adam(model.parameters(), lr=.0002)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
#
#         self.global_step = 0
#
#         for self.epoch in range(0, 30):
#             scheduler.step()
#             for i, batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=64):
#                 image_ids, _, sent_ids, tokens_fixedlen, sent_lengths, _ = batch_data
#
#                 optimizer.zero_grad()
#
#                 sent_ids = np.array(sent_ids).squeeze(1)
#                 sent_lengths = np.array(sent_lengths).squeeze(1)
#                 tokens_fixedlen = np.array(tokens_fixedlen).squeeze(1)
#                 fixed_length = tokens_fixedlen.shape[1]
#
#                 tokens_fixedlen_2 = np.zeros(shape=tokens_fixedlen.shape, dtype=tokens_fixedlen.dtype)
#                 sent_lengths_2 = []
#
#                 for j, image_id in enumerate(image_ids):
#                     caption_item = train_dataset.get_caption_item_by_image_id(image_id)
#                     sents = list(filter(lambda x: x.sentence_id != sent_ids[j], caption_item.sentences))
#                     sent = random.sample(sents, k=1)[0]
#                     tokens = [util.Vocabulary.start_token_id] + sent.token_ids + [util.Vocabulary.end_token_id]
#                     sent_length_2 = min(len(tokens), fixed_length)
#                     sent_lengths_2.append(sent_length_2)
#                     tokens_fixedlen_2[j, :sent_length_2] = tokens[:sent_length_2]
#
#                 tokens_fixedlen = torch.LongTensor(tokens_fixedlen).to(device)
#                 sent_lengths = torch.LongTensor(sent_lengths).to(device)
#                 tokens_fixedlen_2 = torch.LongTensor(tokens_fixedlen_2).to(device)
#                 sent_lengths_2 = torch.LongTensor(sent_lengths_2).to(device)
#
#                 sent1_emb, sent2_emb = model.forward(tokens_fixedlen, sent_lengths, tokens_fixedlen_2, sent_lengths_2)
#                 loss = model.criterion(sent1_emb, sent2_emb)
#
#                 loss.backward()
#                 optimizer.step()
#
#                 loss_scalar = loss.detach().cpu().numpy()
#                 self.writer.add_scalar('loss', loss_scalar, global_step=self.global_step)
#                 self.global_step += 1
#
#             self.save_model(
#                 save_path=os.path.join(self.save_folder, 'saved_models', 'checkpoint_{}'.format(self.epoch)),
#                 state_dict={'model': model.state_dict(),
#                             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
#                             'epoch': self.epoch, 'global_step': self.global_step})
#
#     def test(self, model, test_dataloader):
#         all_sent_emb = []
#         all_sent_id = []
#         for i, batch_data in tqdm(enumerate(test_dataloader)):
#             image_ids, _, sent_ids, tokens_fixedlen, sent_lengths, raw = batch_data
#
#             sent_ids = np.array(sent_ids).squeeze(1)
#             sent_lengths = np.array(sent_lengths).squeeze(1)
#             tokens_fixedlen = np.array(tokens_fixedlen).squeeze(1)
#             fixed_length = tokens_fixedlen.shape[1]
#
#             tokens_fixedlen = torch.LongTensor(tokens_fixedlen).to(device)
#             sent_lengths = torch.LongTensor(sent_lengths).to(device)
#
#             emb = model.forward(tokens_fixedlen, sent_lengths)
#
#             all_sent_emb.extence(emb)
#             all_sent_id.extend(sent_ids)
#
#         assert len(all_sent_emb) == len(all_sent_id)
#         all_sent_emb = np.array(all_sent_emb)   # (n, 512)
#
#         for i in range(len(all_sent_id)):
#             sent_emb = all_sent_emb[i]      # (512,)
#             d = np.dot(sent_emb, all_sent_emb.T)    # (n)
#             pass


class RetrievalPipeline(BasePipeline):
    def __init__(self):
        super().__init__()

    def add_arguments(self, parser):
        super().add_arguments(parser)

    def run(self):
        super().run()
        # self.evaluate()
        self.train()

    def train(self):
        vocab = util.load_custom(r'../data/vocab/vocab_merged_coco_flickr30k.json')

        train_dataset1 = CaptionDataset(dataset_name='coco', vocab=vocab, split='train',
                                       image_mode='none', iter_mode='single', max_sent_length=23)
        # train_dataloader1 = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=512, shuffle=True,
        #                                                           collate_fn=collate_fn_group)

        train_dataset2 = CaptionDataset(dataset_name='flickr30k', vocab=vocab, split='train',
                                       image_mode='none', iter_mode='single', max_sent_length=23)
        # train_dataloader2 = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=512, shuffle=True,
        #                                                           collate_fn=collate_fn_group)

        merged_dataset = torch.utils.data.ConcatDataset([train_dataset1, train_dataset2])
        merged_dataloader = torch.utils.data.dataloader.DataLoader(merged_dataset, batch_size=512, shuffle=True,
                                                                  collate_fn=collate_fn_group)


        model = SentenceMatcher(vocab=vocab, word_embedding_dim=300, hidden_size=512)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=.0002)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

        self.global_step = 0

        for self.epoch in range(0, 10):
            scheduler.step()
            for train_dataloader in [merged_dataloader]:
                for i, batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=64):
                    image_ids, _, sent_ids, tokens_fixedlen, sent_lengths, _ = batch_data

                    optimizer.zero_grad()

                    sent_ids = np.array(sent_ids).reshape(-1)  # (batch_size,)
                    sent_lengths = np.array(sent_lengths)   # (batch_size,)
                    tokens_fixedlen = np.array(tokens_fixedlen)
                    fixed_length = tokens_fixedlen.shape[1]

                    tokens_fixedlen = torch.LongTensor(tokens_fixedlen).to(device)
                    sent_lengths = torch.LongTensor(sent_lengths).to(device)

                    outputs = model.forward(tokens_fixedlen, sent_lengths)
                    loss = util.masked_cross_entropy(outputs, target=tokens_fixedlen[:, 1:].contiguous(), length=sent_lengths - 1)

                    loss.backward()
                    optimizer.step()

                    loss_scalar = loss.detach().cpu().numpy()
                    self.writer.add_scalar('loss', loss_scalar, global_step=self.global_step)
                    self.global_step += 1

            self.save_model(
                save_path=os.path.join(self.save_folder, 'saved_models', 'checkpoint_{}'.format(self.epoch)),
                state_dict={'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                            'epoch': self.epoch, 'global_step': self.global_step})

    def evaluate(self):
        vocab = util.load_custom(r'../data/vocab/vocab_coco.json')

        test_dataset = CaptionDataset(dataset_name='coco', vocab=vocab, split='val',
                                       image_mode='none', iter_mode='single', max_sent_length=18)
        test_dataloader = torch.utils.data.dataloader.DataLoader(test_dataset, batch_size=512, shuffle=True,
                                                                  collate_fn=collate_fn_group)
        model = SentenceMatcher(vocab=vocab, word_embedding_dim=300, hidden_size=512)
        model.to(device)

        state_dict = self.load_model(
            r'/media/wentian/sdb2/work/image_caption_basics/save/2019-05-19_11-18-58_sent_r/saved_models/checkpoint_5')
        model.load_state_dict(state_dict['model'])

        self.test(model, test_dataloader)

    def test(self, model, test_dataloader):
        test_dataset = test_dataloader.dataset

        all_sent_emb = []
        all_sent_id = []
        all_image_id = []   # corresponding image id

        k = 10

        retrieve_result = []

        for i, batch_data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), ncols=64):
            image_ids, _, sent_ids, tokens_fixedlen, sent_lengths, raw = batch_data
            batch_size = len(image_ids)

            sent_ids = np.array(sent_ids).reshape(-1)  # (batch_size,)
            sent_lengths = np.array(sent_lengths)  # (batch_size,)
            tokens_fixedlen = np.array(tokens_fixedlen)
            fixed_length = tokens_fixedlen.shape[1]

            tokens_fixedlen = torch.LongTensor(tokens_fixedlen).to(device)
            sent_lengths = torch.LongTensor(sent_lengths).to(device)

            emb = model.get_sent_embedding(tokens_fixedlen, sent_lengths)
            emb = emb.detach().cpu().numpy()

            all_sent_emb.extend(emb)
            all_sent_id.extend(sent_ids)
            all_image_id.extend(image_ids)

        assert len(all_sent_emb) == len(all_sent_id)
        all_sent_emb = np.array(all_sent_emb)   # (n, 512)
        all_sent_id = np.array(all_sent_id)     #
        all_image_id = np.array(all_image_id)   #

        ranks = {}
        for i in tqdm(range(len(all_sent_id)), total=len(all_sent_id), ncols=64):
            sent_id = all_sent_id[i]
            image_id = all_image_id[i]

            sent_emb = all_sent_emb[i]      # (512,)
            d = np.dot(sent_emb, all_sent_emb.T).flatten()    # (n,)
            sorted_index = np.argsort(d)[::-1]      # reverse the index with [::-1]

            retrieved_sentence_id = all_sent_id[sorted_index]
            retrieved_corr_image_id = all_image_id[sorted_index]

            # rank = np.where(sent_id == retrieved_sentence_id)
            rank = np.where(image_id == retrieved_corr_image_id)[0].min()
            ranks[sent_id] = rank

            retrieve_result.append({'sent_id': sent_id,
                                    'sent': test_dataset.get_sentence_item(sent_id).raw,
                                    'retrieved_sent_id': list(retrieved_sentence_id[:k]),
                                    'retrieved_sents': [test_dataset.get_sentence_item(_).raw for _ in retrieved_sentence_id[:k]]})

        util.dump_custom(retrieve_result, os.path.join(self.save_folder, 'retrieve_result.json'))

        ranks_list = np.array(list(ranks.values()))
        r = [100.0 * len(np.where(ranks_list < i)[0]) / len(ranks) for i in [1, 5, 10]]

        print('recall:', r)


def retrieve_target(model, target_image_id, target_generated_sents, target_sent_id, target_sents, k=10):
        def get_all_sent_embedding(tokens_fixedlen, sent_lengths, batch_size=512):
            assert len(tokens_fixedlen) == len(sent_lengths)
            total = len(tokens_fixedlen)
            print('generating embeddings for {} sents'.format(total))
            all_sent_emb = []

            for batch_index in range(0, len(tokens_fixedlen), batch_size):
                start_index = batch_index * batch_size
                end_index = min(total, (batch_index + 1) * batch_size)
                tokens = tokens_fixedlen[start_index : end_index]
                lengths = sent_lengths[start_index : end_index]

                emb = model.get_sent_embedding(tokens, lengths)
                emb = emb.detach().cpu().numpy()
                all_sent_emb.extend(emb)

            return np.array(all_sent_emb)

        target_image_id = np.array(target_image_id).reshape(-1)
        target_sent_id = np.array(target_sent_id).reshape(-1)

        n_target_image = len(target_image_id)
        target_gen_sents, target_gen_sents_len = target_generated_sents
        assert n_target_image == len(target_gen_sents) and n_target_image == len(target_gen_sents_len)
        target_gen_sents_emb = get_all_sent_embedding(target_gen_sents, target_gen_sents_len)

        n_target_sents = len(target_sent_id)
        target_sents, target_sents_len = target_sents
        assert n_target_sents == len(target_sents) and n_target_sents == len(target_sents_len)
        target_sents_emb = get_all_sent_embedding(target_sents, target_sents_len)

        retrieve_result = []

        for i in tqdm(range(n_target_image)):
            image_id = target_image_id[i]

            gen_sent_emb = target_gen_sents_emb[i]
            d = np.dot(gen_sent_emb, target_sents_emb.T).flatten()    # (n,)
            sorted_index = np.argsort(d)[::-1]

            retrieved_target_sent_id = target_sent_id[sorted_index]
            retrieve_result.append({
                'image_id': image_id,
                'sent_id': retrieved_target_sent_id[:k]
            })

        return retrieve_result



if __name__ == '__main__':
    a = RetrievalPipeline()
    a.run()

