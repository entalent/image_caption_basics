from collections import defaultdict

import torch.utils.data
import torch.utils.data

from util.customjson import *

class ImageItem(JSONSerializable):
    def __init__(self, image_id=None, image_filename=None):
        super().__init__()
        self.image_id, self.image_filename = image_id, image_filename

class SentenceItem(JSONSerializable):
    def __init__(self, sentence_id=None, raw=None, words=None, token_ids=None):
        super().__init__()
        # token_ids can be None
        self.sentence_id, self.raw, self.words, self.token_ids = \
            sentence_id, raw, words, token_ids

class ImageSentencePair(JSONSerializable):
    def __init__(self, image=None, sentence=None, split=None):
        super().__init__()
        self.image, self.sentence, self.split = image, sentence, split

class CaptionItem(JSONSerializable):
    def __init__(self, image=None, sentences=None, split=None):
        super().__init__()
        self.image, self.sentences, self.split = image, sentences, split

JSONSerializable.register_cls(ImageItem, 'II', {'image_id': 'ii', 'image_filename': 'if'})
JSONSerializable.register_cls(SentenceItem, 'SI', {})
JSONSerializable.register_cls(ImageSentencePair, 'ISP', {'image': 'i', 'sentence': 's', 'split': 'sp'})
JSONSerializable.register_cls(CaptionItem, 'CI', {'image': 'i', 'sentences': 'ss', 'split': 'sp'})


class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        dataset_name = kwargs['dataset_name']
        split = kwargs.get('split', 'all')      # default: all
        vocab = kwargs.get('vocab', None)

        self._kwargs = kwargs
        self.dataset_name = dataset_name
        self.split = split
        self.vocab = vocab

        self.image_list = []
        self.sentence_list = []
        self.caption_item_list = []             # list of CaptionItem instance
        self.image_sentence_pair_list = []      # list of ImageSentencePair

        self.caption_item_split = defaultdict(list)
        self.image_sentence_pair_split = defaultdict(list)

        self.load()

        self.image_id_map = dict((image_item.image_id, image_item) for image_item in self.image_list)
        self.sentence_id_map = dict((sentence_item.sentence_id, sentence_item) for sentence_item in self.sentence_list)
        self.image_id_map_2 = dict((caption_item.image.image_id, caption_item) for caption_item in self.caption_item_list)

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def load(self):
        """
        init self.image_list, self.sentence_list, self.caption_item_list, self.image_sentence_pair_list
        :return:
        """
        pass

    def get_image_item(self, image_id):
        return self.image_id_map[int(image_id)]

    def get_sentence_item(self, sentence_id):
        return self.sentence_id_map[int(sentence_id)]

    def get_caption_item_by_image_id(self, image_id):
        return self.image_id_map_2[int(image_id)]


def read_binary_blob(file_name):
    fid = open(file_name, 'rb')

    # s contains size of the blob e.g. num x chanel x length x height x width
    s = np.fromfile(fid, np.int32, 5)

    m = s[0] * s[1] * s[2] * s[3] * s[4]

    # data is the blob binary data in single precision (e.g float in C++)
    data = np.fromfile(fid, np.float32, m)
    data = data.reshape(s)

    fid.close()
    return data


@lru_cache(maxsize=50000)
def load_np(filename):
    return np.load(filename)