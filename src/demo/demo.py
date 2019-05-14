import json
import os
import sys
import time
from functools import lru_cache

sys.path.append(os.getcwd())
sys.path.append(r'/media/wentian/sdb2/work/coco-caption-master')

import torch.utils.data
import h5py
from tqdm import tqdm

import util
import util.reward
from model.model import *


class LSTMLanguageModel(LanguageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_embedding = nn.Linear(in_features=2048, out_features=300)
        self.input_embedding = nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=300, padding_idx=0)
        self.lstm = nn.LSTMCell(input_size=300, hidden_size=512)
        self.output_embedding = nn.Linear(in_features=512, out_features=len(self.vocab))
        self.dropout = nn.Dropout(0.5)

    def prepare_feat(self, input_feature, **kwargs):
        batch_size = len(input_feature)
        prepared_feat = self.image_embedding(input_feature)
        return batch_size, prepared_feat

    def init_state(self, input_feature, **kwargs):
        device = input_feature.device
        batch_size = input_feature.shape[0]
        h_0 = torch.zeros((batch_size, self.lstm.hidden_size)).to(device)
        return self.lstm(input_feature, (h_0, h_0))

    def step(self, input_feature, last_word_id_batch, last_state, **kwargs):
        device = input_feature.device
        last_word_id_batch = torch.LongTensor(np.array(last_word_id_batch).astype(np.int64)).to(device)
        emb = self.input_embedding(last_word_id_batch)
        h, c = self.lstm(emb, last_state)
        output = self.dropout(h)
        output = self.output_embedding(output)
        return output, (h, c), None


custom_collate_fn = lambda x: list(zip(*x))


class CaptionDataset(util.CaptionDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_sent_length = kwargs['max_sent_length']
        self.feat_file = r'/media/wentian/sda1/caption_features/cocotalk_fc/feats_fc.h5'
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
        obj = self.load_json('../data/preprocessed/dataset_coco.json')
        self.caption_item_list = obj['caption_item']

    def __len__(self):
        return len(self.image_sentence_pair_split[self.split])

    def __getitem__(self, index):
        pair = self.image_sentence_pair_split[self.split][index]
        image_item = pair.image
        sent = pair.sentence

        feat = self.read_image_feat(image_item.image_id)

        tokens = [util.Vocabulary.start_token_id] + sent.token_ids + [util.Vocabulary.end_token_id]

        fixed_length = self.max_sent_length + 2     # with <start> and <end>
        sent_length = min(len(tokens), fixed_length)
        tokens_fixedlen = np.zeros((fixed_length,), dtype=np.int64)
        tokens_fixedlen[:sent_length] = tokens[:sent_length]

        # faster, use np.ndarray
        return image_item.image_id, feat, tokens_fixedlen, sent_length, sent.raw

        # to avoid memory leak, use torch.Tensor
        # return image_item.image_id, torch.Tensor(feat), torch.LongTensor(tokens_fixedlen), sent_length, sent.raw


def get_dataloader(dataset_name, split, vocab, **kwargs):
    dataset = CaptionDataset(dataset_name=dataset_name, split=split, vocab=vocab,
                             max_sent_length=kwargs.get('max_sent_length', 17))
    dataloader = torch.utils.data.dataloader.DataLoader(dataset=dataset,
                                                        batch_size=kwargs.get('batch_size', 128),
                                                        shuffle=kwargs.get('shuffle', True),
                                                        num_workers=kwargs.get('num_workers', 0),
                                                        collate_fn=kwargs.get('collate_fn', custom_collate_fn))
    return dataloader


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class DemoPipeline(util.BasePipeline):
    def __init__(self):
        super().__init__()

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument('-saved_model', default='', type=str)

        parser.add_argument('-max_epoch', default=40, type=int)
        parser.add_argument('-sc_after', default=30, type=int)
        parser.add_argument('-max_sample_seq_len', default=25, type=int)    # maximum length during sampling

    def run(self):
        self.train()
        # self.test_data()

    def train(self):
        print('using args:')
        print(json.dumps(self.args.__dict__, indent=4))

        vocab = util.load_custom('../data/vocab/vocab_coco.json')
        train_dataloader = get_dataloader('coco', 'train', vocab)
        test_dataloader = get_dataloader('coco', 'test', vocab, batch_size=1, shuffle=False)

        model = LSTMLanguageModel(vocab=vocab)
        model.to(device)
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 5e-4}])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3,
                                                    gamma=0.8)

        self.epoch = 0
        self.global_step = 0

        if len(self.args.saved_model) > 0:
            state_dict = self.load_model('/media/wentian/sdb2/work/image_caption_basics/save/2019-05-13_22-44-50/saved_models/checkpoint_4')
            model.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])
            scheduler.load_state_dict(state_dict['scheduler'])
            self.epoch = state_dict['epoch']
            self.global_step = state_dict['global_step']
            print('starting at epoch {}, global step {}'.format(self.epoch, self.global_step))

        max_sample_seq_len = self.args.max_sample_seq_len
        max_epoch = self.args.max_epoch
        sc_after = self.args.sc_after

        while self.epoch < max_epoch:
            scheduler.step()
            model.train(True)
            print('epoch {}'.format(self.epoch))

            self_critical_flag = self.epoch >= sc_after
            if self_critical_flag:
                util.reward.init_scorer(df='/media/wentian/sdb2/work/image_caption_basics/data/preprocessed/ngram_coco_train_words')

            for i, batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=64):
                image_id, feat, tokens_fixedlen, sent_length, raw = batch_data
                feat = torch.Tensor(np.array(feat)).to(device)
                tokens = torch.LongTensor(np.array(tokens_fixedlen)).to(device)
                sent_length = torch.LongTensor(sent_length)

                optimizer.zero_grad()

                if not self_critical_flag:
                    # mask = util.get_word_mask(tokens.shape, sent_length)
                    token_input = tokens[:, :-1].contiguous()
                    token_target = tokens[:, 1:].contiguous()

                    outputs = model.forward(input_feature=feat, input_sentence=token_input)

                    loss = util.masked_cross_entropy(outputs, token_target, sent_length - 1)
                    loss.backward()
                    optimizer.step()

                    loss_scalar = loss.detach().cpu()
                    self.writer.add_scalar('loss/cross_entropy', loss_scalar, global_step=self.global_step)
                else:
                    start_time = time.time()
                    sample_logprob, sample_seq, _ = model.sample(input_feature=feat, max_length=max_sample_seq_len,
                                                                 sample_max=False)

                    sample_time = time.time() - start_time; start_time = time.time()

                    with torch.no_grad():
                        model.train(False)
                        greedy_logprob, greedy_seq, _ = model.sample(input_feature=feat, max_length=max_sample_seq_len,
                                                                     sample_max=True)
                        model.train(True)

                    greedy_time = time.time() - start_time; start_time = time.time()

                    gts_raw = [
                        [s.raw + ' ' + util.Vocabulary.end_token
                        for s in train_dataloader.dataset.get_caption_item_by_image_id(id).sentences]
                        for id in image_id
                    ]

                    reward = util.reward.get_self_critical_reward(sample_seq, greedy_seq, gts_raw,
                                                         weights={'bleu': 0.5, 'cider': 0.5}, vocab=vocab)
                    loss = util.reward.rl_criterion(log_prob=sample_logprob, generated_seq=sample_seq, reward=reward)

                    compute_loss_time = time.time() - start_time; start_time = time.time()

                    loss.backward()
                    optimizer.step()

                    if self.global_step % 20 == 0:
                        all_lr = {}
                        for i, param_group in enumerate(optimizer.param_groups):
                            all_lr[str(i)] = param_group['lr']
                        self.writer.add_scalars('learning_rate', all_lr, global_step=self.global_step)
                        loss_scalar = loss.detach().cpu()
                        self.writer.add_scalar('loss/self_critical', loss_scalar, global_step=self.global_step)
                        backward_time = time.time() - start_time
                        self.writer.add_scalars('sc_time', {'sample': sample_time, 'greedy': greedy_time,
                                                            'compute_loss': compute_loss_time, 'backward': backward_time}, self.global_step)


                self.global_step += 1

            self.test(test_dataloader, model, vocab)

            self.epoch += 1
            self.save_model(save_path=os.path.join(self.save_folder, 'saved_models', 'checkpoint_{}'.format(self.epoch)),
                            state_dict={'model': model.state_dict(),
                                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                                        'epoch': self.epoch, 'global_step': self.global_step})



    def test(self, test_dataloader, model, vocab):
        result_generator = util.COCOResultGenerator()
        model.train(False)
        for i, batch_data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), ncols=64):
            image_id, feat, tokens_fixedlen, sent_length, raw = batch_data

            image_id = image_id[0]
            result_generator.add_annotation(image_id, raw[0])
            if result_generator.has_output(image_id):
                continue

            feat = torch.Tensor(np.array(feat)).to(device)
            log_prob_seq, word_id_seq, _ = model.sample_beam(input_feature=feat, max_length=20, beam_size=5)

            words = util.trim_generated_tokens(word_id_seq)
            words = [vocab.get_word(i) for i in words]
            sent = ' '.join(words)
            result_generator.add_output(image_id, sent, [word_id_seq, log_prob_seq])

            # print(word_id_seq, sent)

        ann_file = os.path.join(self.save_folder, 'annotation.json')
        result_file = os.path.join(self.save_folder, 'result_{}.json'.format(self.epoch))
        metric_file = os.path.join(self.save_folder, 'metric_{}.json'.format(self.epoch))
        result_generator.dump_annotation_and_output(ann_file, result_file)

        # eval_cmd = '{} /home/wentian/work/coco-caption-master/eval_eng.py {} {} {}'\
        #     .format(sys.executable, ann_file, result_file, metric_file)
        # os.system(eval_cmd)

        metrics = util.eval(ann_file, result_file)
        self.writer.add_scalars(main_tag='metric/', tag_scalar_dict=metrics, global_step=self.global_step)


    def test_data(self):
        print('testing dataloader...')
        vocab = util.load_custom('../data/vocab/vocab_coco.json')
        train_dataloader = get_dataloader('coco', 'train', vocab, num_workers=1, collate_fn=lambda x: x)
        for i, batch_data in tqdm(enumerate(train_dataloader), ncols=64, total=len(train_dataloader)):
            pass


def main():
    p = DemoPipeline()
    p.run()

if __name__ == '__main__':
    main()