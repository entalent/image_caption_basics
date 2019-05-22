import json
import os
import pickle
import sys
import time
from functools import lru_cache

sys.path.append(os.getcwd())
sys.path.append('./src')
sys.path.append(r'/media/wentian/sdb2/work/coco-caption-master')

import torch.utils.data
import torch.nn as nn
import h5py
from tqdm import tqdm
import numpy as np
import util
import util.reward
from model.model import LanguageModel

from FCModel import FCModel


class CaptionDataset(util.CaptionDataset):
    def __init__(self, **kwargs):
        self.iter_mode = kwargs.get('iter_mode', 'single')
        assert self.iter_mode in ['single', 'group']
        self.max_sent_length = kwargs['max_sent_length']
        self.image_mode = kwargs.get('image_mode', 'fc_feat')
        self.use_restval = kwargs.get('use_restval', True)

        super().__init__(**kwargs)

        if self.dataset_name == 'coco':
            self.feat_file = r'/media/wentian/nvme0n1p5/work/coco_fc.h5'
        elif self.dataset_name == 'flickr30k':
            self.feat_file = r'/media/wentian/nvme0n1p5/work/flickr30k_fc.h5'

        for sent in self.sentence_list:
            sent.token_ids = [self.vocab.get_index(w) for w in sent.words]

    def read_image_feat(self, image_id):
        self.file_avg = h5py.File(self.feat_file, 'r', libver='latest', swmr=True)
        feat = np.array(self.file_avg[str(image_id)]).astype(np.float32)
        return feat

    @staticmethod
    @lru_cache(maxsize=5)
    def load_json(filename):
        return util.load_custom(filename)

    def load(self):
        # with open('../data/preprocessed/dataset_coco.json', 'r') as f:
        #     self.caption_item_list = util.load_custom(f)['caption_item']
        dataset_file = '../data/preprocessed/dataset_{}.json'.format(self.dataset_name)
        obj = self.load_json(dataset_file)
        caption_item_list = obj['caption_item']
        if self.use_restval:
            for caption_item in caption_item_list:
                if caption_item.split == 'restval':
                    caption_item.split = 'train'
        self.caption_item_list = obj['caption_item']

    def __len__(self):
        if self.iter_mode == 'single':
            return len(self.image_sentence_pair_split[self.split])
        elif self.iter_mode == 'group':
            return len(self.caption_item_split[self.split])

    def __getitem__(self, index):
        if self.iter_mode == 'single':
            pair = self.image_sentence_pair_split[self.split][index]
            image_item = pair.image
            sents = [pair.sentence]
        else:
            i = self.caption_item_split[self.split][index]
            image_item = i.image
            sents = i.sentences

        fixed_length = self.max_sent_length + 2  # with <start> and <end>

        # if self.dataset_name == 'coco':
        #     feat = self.read_image_feat(image_item.image_id)
        # else:
        #     feat = self.read_image_feat(image_item.image_filename)
        if self.image_mode == 'fc_feat':
            feat = self.read_image_feat(image_item.image_filename)
        elif self.image_mode == 'none':
            feat = torch.Tensor([0])

        all_tokens = np.zeros((len(sents), fixed_length), dtype=np.int64)
        sent_lengths = []
        sent_ids = []
        for i, sent in enumerate(sents):
            tokens = [util.Vocabulary.start_token_id] + sent.token_ids + [util.Vocabulary.end_token_id]
            sent_length = min(len(tokens), fixed_length)
            sent_lengths.append(sent_length)
            all_tokens[i, :sent_length] = tokens[:sent_length]
            sent_ids.append(sent.sentence_id)

        # faster, use np.ndarray
        return image_item.image_id, feat, sent_ids, all_tokens, sent_lengths, [sent.raw for sent in sents]

        # to avoid memory leak, use torch.Tensor
        # return image_item.image_id, torch.Tensor(feat), torch.LongTensor(tokens_fixedlen), sent_length, sent.raw


class Args:
    def __init__(self, d):
        for k, v in d.items():
            self.__setattr__(k, v)


# wrapper class to use the FCModel from https://github.com/ruotianluo/self-critical.pytorch
class FCModelWrapper(nn.Module):
    def __init__(self, vocab_size, seq_length):
        super().__init__()
        opt_dict = {
            'vocab_size': vocab_size,
            'input_encoding_size': 512,
            'rnn_type': 'lstm',
            'rnn_size': 512,
            'num_layers': 1,
            'drop_prob_lm': 0.5,
            'seq_length': seq_length,
            'fc_feat_size': 2048
        }
        opt = Args(opt_dict)

        self.model = FCModel(opt)

    def forward(self, input_feature, input_sentence, ss_prob):
        fc_feats = input_feature
        att_feats = None
        seq = input_sentence
        self.model.ss_prob = ss_prob
        return self.model._forward(fc_feats=fc_feats, att_feats=att_feats, seq=seq)

    def sample(self, input_feature, max_length, sample_max=True):
        _opt = {'sample_max': sample_max}
        seq, seqLogProbs = self.model._sample(fc_feats=input_feature, att_feats=None, opt=_opt)
        seq = seq.detach().cpu().numpy()
        return seqLogProbs, seq, None

    def sample_beam(self, input_feature, max_length=20, beam_size=5):
        _opt = {'beam_size': beam_size}
        seq, seqLogProbs = self.model._sample_beam(fc_feats=input_feature, att_feats=None, opt=_opt)
        seq = seq.detach().cpu().numpy()
        return seqLogProbs, seq, None


# loss from https://github.com/ruotianluo/self-critical.pytorch
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


def get_dataloader(dataset_name, split, vocab, **kwargs):
    dataset = CaptionDataset(dataset_name=dataset_name, split=split, vocab=vocab,
                             max_sent_length=kwargs.get('max_sent_length', 18), iter_mode=kwargs.get('iter_mode', 'single'))
    dataloader = torch.utils.data.dataloader.DataLoader(dataset=dataset,
                                                        batch_size=kwargs.get('batch_size', 128),
                                                        shuffle=kwargs.get('shuffle', True),
                                                        num_workers=kwargs.get('num_workers', 0),
                                                        collate_fn=collate_fn_group)
    return dataloader


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def collate_fn_group(data):
    data = list(zip(*data))

    image_ids, feats, sent_ids, tokens_fixedlen, sent_lengths, raw = data
    sent_per_image = [len(i) for i in sent_ids]

    if isinstance(feats[0], torch.Tensor):
        _feats = []
        for i, num_sent in enumerate(sent_per_image):
            _f = feats[i].unsqueeze(0)      # (1, 2048)
            _f = _f.expand(num_sent, *_f.shape[1:])     # (n, 2048)
            _feats.append(_f)
        feats = torch.cat(_feats, dim=0)
    elif isinstance(feats[0], np.ndarray):
        _feats = []
        for i, num_sent in enumerate(sent_per_image):
            _feats.append(np.repeat(np.expand_dims(feats[i], 0), num_sent, axis=0))
        feats = np.concatenate(_feats, axis=0)

    if isinstance(tokens_fixedlen[0], torch.Tensor):
        tokens_fixedlen = torch.cat(tokens_fixedlen, dim=0)
    else:
        tokens_fixedlen = np.concatenate(tokens_fixedlen, axis=0)

    if isinstance(sent_lengths[0], torch.Tensor):
        sent_lengths = torch.cat(sent_lengths, dim=0)
    else:
        sent_lengths = np.concatenate(sent_lengths, axis=0)

    """
    image_ids: list, (batch_size,)
    feats: tensor, (total_sents, feat_dim)
    sent_ids: list, ([1, 2, 3], [4, 5], ...)
    tokens_fixedlen: tensor, (total_sents, max_len)
    sent_lengths: tensor, (total_sents)
    raw: list, (['sent 1', 'sent 2', 'sent 3'], ['sent 4', 'sent 5'], ...)
    """
    return image_ids, feats, sent_ids, tokens_fixedlen, sent_lengths, raw


class DemoPipeline(util.BasePipeline):
    def __init__(self):
        super().__init__()

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument('action', default='train', type=str)
        parser.add_argument('-dataset', default='coco', type=str)
        parser.add_argument('-saved_model', default='', type=str)

        parser.add_argument('-max_epoch', default=40, type=int)
        parser.add_argument('-sc_after', default=30, type=int)
        parser.add_argument('-max_sample_seq_len', default=16, type=int)    # maximum length during sampling
        parser.add_argument('-max_sent_length', default=16, type=int)       # maximum length of provided sentence

        parser.add_argument('-grad_clip', default=0.1, type=float)

        parser.add_argument('-ss_start', default=0, type=int)
        parser.add_argument('-ss_increase_every', default=5, type=int)
        parser.add_argument('-ss_prob_increase', default=0.05, type=int)
        parser.add_argument('-ss_prob_max', default=0.25, type=int)

        parser.add_argument('-beam_size', default=1, type=int)  # beam size during test


    def run(self):
        print('using args:')
        print(json.dumps(self.args.__dict__, indent=4))

        if self.args.action == 'train':
            self.train()
        elif self.args.action == 'test':
            self.evaluate()

    def train(self):
        dataset_name = self.args.dataset

        # vocab = util.load_custom('../data/vocab/vocab_{}.json'.format(dataset_name))
        # merged vocab
        vocab_path = '../data/vocab/vocab_coco.json'
        print('loading vocab from {}'.format(vocab_path))
        vocab = util.load_custom(vocab_path)
        # pretrained_embedding = util.read_glove_embedding(vocab)

        train_dataloader = get_dataloader(dataset_name, 'train', vocab, batch_size=10, max_sent_length=self.args.max_sent_length, iter_mode='group')
        test_dataloader = get_dataloader(dataset_name, 'val', vocab, batch_size=10, shuffle=False, iter_mode='group')

        # model = LSTMLanguageModel(vocab=vocab, pretrained_embedding=pretrained_embedding)
        model = FCModelWrapper(vocab_size=len(vocab), seq_length=self.args.max_sent_length)
        model.to(device)
        self.crit = LanguageModelCriterion()
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 5e-4}])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3,
                                                    gamma=0.8)

        self.epoch = 0
        self.global_step = 0

        if len(self.args.saved_model) > 0:
            state_dict = self.load_model(self.args.saved_model)
            model.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])
            scheduler.load_state_dict(state_dict['scheduler'])
            self.epoch = state_dict['epoch']
            self.global_step = state_dict['global_step']
            print('starting at epoch {}, global step {}'.format(self.epoch, self.global_step))

        max_sent_length = self.args.max_sent_length
        max_sample_seq_len = self.args.max_sample_seq_len
        max_epoch = self.args.max_epoch
        sc_after = self.args.sc_after

        ss_start, ss_increase_every, ss_prob_increase, ss_prob_max = \
            self.args.ss_start, self.args.ss_increase_every, self.args.ss_prob_increase, self.args.ss_prob_max

        while self.epoch < max_epoch:
            scheduler.step()
            model.train(True)

            ss_prob = 0
            if ss_start >= 0:
                ss_prob = min(((self.epoch - ss_start) // ss_increase_every) * ss_prob_increase, ss_prob_max)

            print('epoch {}'.format(self.epoch), 'using ss_prob {}'.format(ss_prob))

            self_critical_flag = (self.epoch >= sc_after and sc_after >= 0)
            if self_critical_flag:
                util.reward.init_scorer(df='/media/wentian/sdb2/work/image_caption_basics/data/preprocessed/ngram_coco_train_words')

            for i, batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=64):
                image_id, feat, sent_ids, tokens_fixedlen, sent_length, raw = batch_data

                feat = torch.Tensor(np.array(feat)).to(device)
                tokens = torch.LongTensor(np.array(tokens_fixedlen)).to(device)  # (batch_size, sent_per_img, max_len) -> (batch_size, max_len)
                sent_length = torch.LongTensor(sent_length)

                optimizer.zero_grad()

                add_log = self.global_step % 20 == 0

                if not self_critical_flag:
                    mask = util.get_word_mask(tokens.shape, sent_length)
                    token_input = tokens
                    token_target = tokens[:, 1:].contiguous()

                    outputs = model.forward(input_feature=feat, input_sentence=token_input, ss_prob=ss_prob)

                    mask = torch.FloatTensor(mask).to(outputs.device)
                    loss = self.crit(outputs, token_target, mask[:, 1:])

                    loss.backward()
                    util.clip_gradient(optimizer, self.args.grad_clip)
                    optimizer.step()

                    if add_log:
                        loss_scalar = loss.detach().cpu()
                        self.writer.add_scalar('loss/cross_entropy', loss_scalar, global_step=self.global_step)
                else:
                    start_time = time.time()
                    sample_logprob, sample_seq, _ = model.sample(input_feature=feat, max_length=max_sample_seq_len + 1,
                                                                 sample_max=False)

                    sample_time = time.time() - start_time; start_time = time.time()

                    model.train(False)
                    with torch.no_grad():
                        greedy_logprob, greedy_seq, _ = model.sample(input_feature=feat, max_length=max_sample_seq_len + 1,
                                                                     sample_max=True)
                    model.train(True)

                    greedy_time = time.time() - start_time; start_time = time.time()

                    # gts_raw = [
                    #     [' '.join(s.words[:max_sent_length] + [util.Vocabulary.end_token])
                    #     for s in train_dataloader.dataset.get_caption_item_by_image_id(id).sentences]
                    #     for id in image_id
                    # ]
                    gts_raw = []
                    for id in image_id:
                        g = []
                        for s in train_dataloader.dataset.get_caption_item_by_image_id(id).sentences:
                            if len(s.words) < max_sent_length:
                                g.append(' '.join(s.words + [util.Vocabulary.end_token]))
                            else:
                                g.append(' '.join(s.words[:max_sent_length]))
                        gts_raw.append(g)
                    gts_raw_expanded = []
                    for _i, _image_id in enumerate(image_id):
                        n_sent = len(sent_ids[_i])
                        gts_raw_expanded.extend([gts_raw[_i]] * n_sent)

                    reward = util.reward.get_self_critical_reward(sample_seq, greedy_seq, gts_raw_expanded,
                                                                  weights={'bleu': 0, 'cider': 1.0}, vocab=vocab)
                    # reward = util.reward.get_self_critical_reward(sample_seq, greedy_seq, gts_raw,
                    #                                               weights={'bleu': 1.0, 'cider': 0.0}, vocab=vocab)
                    loss = util.reward.rl_criterion(log_prob=sample_logprob, generated_seq=sample_seq, reward=reward)

                    compute_loss_time = time.time() - start_time; start_time = time.time()

                    loss.backward()
                    util.clip_gradient(optimizer, self.args.grad_clip)
                    optimizer.step()

                    if add_log:
                        avg_reward = np.mean(reward[:, 0])
                        self.writer.add_scalar('average_reward', avg_reward, global_step=self.global_step)
                        loss_scalar = loss.detach().cpu()
                        self.writer.add_scalar('loss/self_critical', loss_scalar, global_step=self.global_step)
                        backward_time = time.time() - start_time
                        self.writer.add_scalars('sc_time', {'sample': sample_time, 'greedy': greedy_time,
                                                            'compute_loss': compute_loss_time, 'backward': backward_time}, self.global_step)
                if add_log:
                    self.writer.add_scalar('scheduled_sampling_prob', ss_prob, global_step=self.global_step)
                    all_lr = {}
                    for i, param_group in enumerate(optimizer.param_groups):
                        all_lr[str(i)] = param_group['lr']
                    self.writer.add_scalars('learning_rate', all_lr, global_step=self.global_step)

                self.global_step += 1

            # if self_critical_flag or (self.epoch % 5 == 0) or (self.epoch + 1 == sc_after and sc_after >= 0):
            self.test(test_dataloader, model, vocab)

            self.save_model(save_path=os.path.join(self.save_folder, 'saved_models', 'checkpoint_{}'.format(self.epoch)),
                            state_dict={'model': model.state_dict(),
                                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                                        'epoch': self.epoch, 'global_step': self.global_step})
            self.epoch += 1


    def test(self, test_dataloader, model, vocab):
        result_generator = util.COCOResultGenerator()
        model.train(False)
        beam_size = self.args.beam_size

        for i, batch_data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), ncols=64):
            image_ids, feats, sent_ids, _, _, raw = batch_data
            batch_size = len(image_ids)
            # feats = torch.Tensor(feats).to(device)

            feat_index = 0
            for batch_index in range(batch_size):
                image_id = image_ids[batch_index]
                for sent in raw[batch_index]:
                    result_generator.add_annotation(image_id, sent)

                feat = torch.Tensor([np.array(feats[feat_index])]).to(device)
                feat_index += len(sent_ids[batch_index])

                # feat = feats[batch_index].unsqueeze(0)
                log_prob_seq, word_id_seq, _ = model.sample(input_feature=feat,
                                                            max_length=self.args.max_sample_seq_len + 1)

                word_id_seq = word_id_seq.reshape(-1)
                log_prob_seq = log_prob_seq.detach().cpu().numpy()
                words = util.trim_generated_tokens(word_id_seq)
                words = [vocab.get_word(i) for i in words]
                sent = ' '.join(words)
                result_generator.add_output(image_id, sent, metadata=None)
                                            # metadata={'word_id': word_id_seq,
                                            #           'log_prob_seq': log_prob_seq,
                                            #           'perplexity': util.calc_perplexity(log_prob_seq)})

            # print(word_id_seq, sent)

        ann_file = os.path.join(self.save_folder, 'annotation.json')
        result_file = os.path.join(self.save_folder, 'result_{}.json'.format(self.epoch))
        metric_file = os.path.join(self.save_folder, 'metric_{}.json'.format(self.epoch))
        result_generator.dump_annotation_and_output(ann_file, result_file)

        metrics = util.eval(ann_file, result_file)
        self.writer.add_scalars(main_tag='metric/', tag_scalar_dict=metrics, global_step=self.global_step)


def main():
    p = DemoPipeline()
    p.run()

if __name__ == '__main__':
    main()