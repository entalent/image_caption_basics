import os
import pickle
import sys
import json
import re
from util.data import CaptionItem, SentenceItem, ImageItem
from util.preprocess import preprocess_captions, preprocess_ngrams
from util.customjson import *

annotation_coco = r'/media/wentian/sdb2/work/caption_dataset/caption_datasets/dataset_coco.json'
vocab_save_path = r'../data/vocab'
preprocessed_dataset_path = r'../data/preprocessed'

if not os.path.exists(vocab_save_path):
    os.makedirs(vocab_save_path)
if not os.path.exists(preprocessed_dataset_path):
    os.makedirs(preprocessed_dataset_path)

def preprocess_coco():
    with open(annotation_coco, 'r') as f:
        annotation = json.load(f)

    caption_item_list = []

    all_sent_item = []

    for img in annotation['images']:
        image_filename = img['filename']
        split = img['split']
        image_id = int(re.split(r'[_|.]', image_filename)[-2])

        image_item = ImageItem(image_id=image_id, image_filename=image_filename)
        sent_list = []

        for sent in img['sentences']:
            sent_item = SentenceItem(sentence_id=sent['sentid'], raw=sent['raw'], words=None, token_ids=None)
            sent_list.append(sent_item)
            all_sent_item.append(sent_item)

        caption_item_list.append(CaptionItem(image=image_item, sentences=sent_list, split=split))

    tokenized_caption_list, vocab_filtered = preprocess_captions('coco', caption_list=[s.raw for s in all_sent_item])

    for i, sent_item in enumerate(all_sent_item):
        sent_item.words = tokenized_caption_list[i]

    vocab_file = os.path.join(vocab_save_path, 'vocab_coco.json')
    dump_custom(vocab_filtered, vocab_file)
    print('saved vocab to {}'.format(vocab_file))

    dataset_file = os.path.join(preprocessed_dataset_path, 'dataset_coco.json')
    dump_custom({'dataset': 'coco', 'caption_item': caption_item_list}, dataset_file)
    print('saved preprocessed dataset to {}'.format(dataset_file))

    df_split = 'train'
    df_data = preprocess_ngrams(caption_item_list, split=df_split, vocab=vocab_filtered)
    ngram_file = os.path.join(preprocessed_dataset_path, 'ngram_coco_{}_words.pkl'.format(df_split))
    with open(ngram_file, 'wb') as f:
        pickle.dump(df_data, f)
    print('preprocessed ngram, dumped to {}'.format(ngram_file))


if __name__ == '__main__':
    preprocess_coco()
