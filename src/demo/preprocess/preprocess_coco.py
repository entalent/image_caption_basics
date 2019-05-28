import copy
import os
import pickle
import string
import sys
import json
import re
from collections import Counter

from util import Vocabulary
from util.data import CaptionItem, SentenceItem, ImageItem
from util.preprocess import preprocess_captions, preprocess_ngrams
from util.customjson import *

annotation_coco = r'/media/wentian/sdb2/work/caption_dataset/caption_datasets/dataset_coco.json'
annotation_flickr30k = r'/media/wentian/sdb2/work/caption_dataset/caption_datasets/dataset_flickr30k.json'

oxford102_annotation_path = r'/media/wentian/sdb2/work/caption_dataset/cvpr2016_flowers'
cub200_annotation_path = r'/media/wentian/sdb2/work/caption_dataset/cvpr2016_cub'
split_files = {'train': 'trainclasses.txt', 'val': 'valclasses.txt', 'test': 'testclasses.txt'}
text_c10_path = 'text_c10'

data_path = '../data'
vocab_save_path = os.path.join(data_path, 'vocab')
preprocessed_dataset_path = os.path.join(data_path, 'preprocessed')

if not os.path.exists(vocab_save_path):
    os.makedirs(vocab_save_path)
if not os.path.exists(preprocessed_dataset_path):
    os.makedirs(preprocessed_dataset_path)


def preprocess_prepared_dataset(dataset_name='coco'):
    if dataset_name == 'coco':
        with open(annotation_coco, 'r') as f:
            annotation = json.load(f)
    elif dataset_name == 'flickr30k':
        with open(annotation_flickr30k, 'r') as f:
            annotation = json.load(f)
    else:
        return

    caption_item_list = []

    all_split = Counter()
    for img in annotation['images']:
        image_filename = img['filename']
        split = img['split']
        all_split.update([split])
        image_id = int(re.split(r'[_|.]', image_filename)[-2])

        image_item = ImageItem(image_id=image_id, image_filename=image_filename)
        sent_list = []

        for sent in img['sentences']:
            sent_item = SentenceItem(sentence_id=sent['sentid'], raw=sent['raw'], words=None, token_ids=None)
            sent_list.append(sent_item)

        caption_item_list.append(CaptionItem(image=image_item, sentences=sent_list, split=split))

    print('splits:', all_split)

    save_preprocessed_dataset(dataset_name, caption_item_list)


_table = str.maketrans(dict.fromkeys(string.punctuation))


def preprocess_fine_grained(dataset_name='oxford102'):
    def _check_illegal_char(filename, sentence):
        flag = False
        for i, c in enumerate(sentence):
            if ord(c) not in range(128):
                print('file {}\nillegal character: {} at {}'.format(filename, ord(c), i))
                flag = True
                break
        if flag:
            s = ''.join([c if ord(c) in range(128) else ' ' for c in sentence])
            s = ' '.join(s.split())
            print(sentence)
            print(s)
            return s
        else:
            return sentence

    if dataset_name == 'oxford102':
        ann_path = oxford102_annotation_path
    elif dataset_name == 'cub200':
        ann_path = cub200_annotation_path
    else:
        return

    class_split = {}
    for split in split_files:
        split_file = os.path.join(ann_path, split_files[split])
        with open(split_file, 'r') as f:
            for line in f:
                class_name = line.strip()
                class_split[class_name] = split

    all_caption_items = []
    image_id_counter = 0
    sentence_id_counter = 0
    for class_name, split in class_split.items():
        class_index = re.findall(r'\d+', class_name)[0]
        class_index = str(class_index)
        _ = os.path.join(ann_path, text_c10_path)
        class_folder_name = list(filter(lambda x: class_index in x and os.path.isdir(os.path.join(_, x)), os.listdir(_)))[0]
        class_folder = os.path.join(ann_path, text_c10_path, class_folder_name)

        print('class index:', class_index, class_name, split, '->', class_folder)

        img_anns = filter(lambda x: x.endswith('.txt'), os.listdir(class_folder))
        for img_ann_filename in img_anns:
            image_name = img_ann_filename.split('.')[0]
            image_filename = image_name + '.jpg'
            image_id = image_id_counter; image_id_counter += 1

            image_item = ImageItem(image_id=image_id, image_filename=image_filename)

            sentences = []
            with open(os.path.join(class_folder, img_ann_filename), 'r') as f:
                for line in f:
                    raw = line.strip().translate(_table)
                    raw = _check_illegal_char(img_ann_filename, raw)
                    sentence_id = sentence_id_counter; sentence_id_counter += 1
                    sentence_item = SentenceItem(sentence_id=sentence_id, raw=raw, words=None, token_ids=None)
                    sentences.append(sentence_item)

            all_caption_items.append(CaptionItem(image=image_item, sentences=sentences, split=split))

    save_preprocessed_dataset(dataset_name, all_caption_items)


def save_preprocessed_dataset(dataset_name, all_caption_items):
    all_sentence_items = []
    for caption_item in all_caption_items:
        all_sentence_items.extend(caption_item.sentences)

    print('generating vocab and ngrams for dataset {}'.format(dataset_name))
    tokenized_caption_list, vocab_filtered = preprocess_captions(dataset_name,
                                                                 caption_list=[s.raw for s in all_sentence_items])

    for i, sent_item in enumerate(all_sentence_items):
        sent_item.words = tokenized_caption_list[i]

    vocab_file = os.path.join(vocab_save_path, 'vocab_{}.json'.format(dataset_name))
    dump_custom(vocab_filtered, vocab_file)
    print('saved vocab to {}, total {} words'.format(vocab_file, len(vocab_filtered)))

    dataset_file = os.path.join(preprocessed_dataset_path, 'dataset_{}.json'.format(dataset_name))
    print('total {} images'.format(len(all_caption_items)))
    dump_custom({'dataset': dataset_name, 'caption_item': all_caption_items}, dataset_file)
    print('saved preprocessed dataset to {}'.format(dataset_file))

    df_split = 'train'
    print('preprocessing ngrams...')
    df_data = preprocess_ngrams(all_caption_items, split=df_split, vocab=vocab_filtered)
    ngram_file = os.path.join(preprocessed_dataset_path, 'ngram_{}_{}_words.p'.format(dataset_name, df_split))
    with open(ngram_file, 'wb') as f:
        pickle.dump(df_data, f)
    print('preprocessed ngram, dumped to {}'.format(ngram_file))


def _merge_vocab(vocab_src, vocab_target):
    for i, vocab in enumerate([vocab_src, vocab_target]):
        print('vocab {} has {} words'.format(i, len(vocab)))

    vocab_merged = copy.deepcopy(vocab_src)
    print('{} words before merge'.format(len(vocab_merged)))
    for w in vocab_target.word2idx:
        if w not in vocab_merged.word2idx:
            vocab_merged._add_word(w)
    print('{} words after merge'.format(len(vocab_merged)))
    return vocab_merged


if __name__ == '__main__':
    preprocess_prepared_dataset('coco')
    preprocess_prepared_dataset('flickr30k')
    preprocess_fine_grained('cub200')
    preprocess_fine_grained('oxford102')

    for target in 'flickr30k', 'cub200', 'oxford102':
        vocab_coco = load_custom(os.path.join(vocab_save_path, 'vocab_coco.json'))
        vocab_target = load_custom(os.path.join(vocab_save_path, 'vocab_{}.json'.format(target)))
        vocab_merged = _merge_vocab(vocab_coco, vocab_target)
        dump_custom(vocab_merged, os.path.join(vocab_save_path, 'vocab_merged_coco_{}.json'.format(target)))
        print('vocabulary of coco has {} words'.format(len(vocab_coco)))
        print('vocabulary of {} has {} words, merged vocabulary has {} words'.format(target, len(vocab_target),
                                                                                     len(vocab_merged)))
