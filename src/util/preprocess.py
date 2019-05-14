import os
import string
import sys
from collections import Counter, defaultdict

from .vocabulary import Vocabulary

xrange = range


_table = str.maketrans(dict.fromkeys(string.punctuation))

def default_tokenize_func(sent):
    return str(sent).strip().lower().translate(_table).split()


def preprocess_captions(dataset_name, caption_list, word_count_th=5, tokenize_func=default_tokenize_func):
    print('preprocessing dataset {}'.format(dataset_name))
    word_counter = Counter()
    sentence_length_counter = Counter()

    # tokenize
    tokenized_caption_list = []
    for i, sentence in enumerate(caption_list):
        tokens = tokenize_func(sentence)
        sentence_length_counter.update([len(tokens)])
        word_counter.update(tokens)
        tokenized_caption_list.append(tokens)

    # build vocab
    print('full vocab has {} words'.format(len(word_counter)))
    vocab_filtered = Vocabulary()
    valid_words = []
    total_word_count, unk_word_count = 0, 0
    for word, count in word_counter.items():
        total_word_count += count
        if count >= word_count_th:
            valid_words.append(word)
        else:
            unk_word_count += count
    valid_words.sort()
    for w in valid_words:
        vocab_filtered._add_word(w)
    print('filtered vocab has {} words'.format(len(vocab_filtered)))
    print('total {} words in all sentences, {} word is <unk>'.format(total_word_count, unk_word_count))

    # sentence length stats
    c = list(sentence_length_counter.items())
    c.sort(key=lambda x: x[0])
    s = 0
    for sent_length, count in c:
        s += count
        percent = (float(s) / len(caption_list))
        print('length <= {} has {} ({:.2f}%) sentences'.format(sent_length, s, percent * 100.0))

    return tokenized_caption_list, vocab_filtered


def compute_doc_freq(crefs):
    '''
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    '''
    document_frequency = defaultdict(float)
    for refs in crefs:
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
            document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
    return document_frequency


def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in xrange(1, n + 1):
        for i in xrange(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):  ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]


def create_crefs(refs):
    crefs = []
    for ref in refs:
        # ref is a list of 5 captions
        crefs.append(cook_refs(ref))
    return crefs


def preprocess_ngrams(caption_items, split, vocab):
    """

    :param caption_items: list of CaptionItem
    :param split: one of 'train', 'val', 'test', 'all'
    :return: {'document_frequency': df, 'ref_len': ref_len}
    """
    count_imgs = 0

    refs_words = []

    for caption_item in caption_items:
        if split == caption_item.split or split == 'all':
            ref_words = []
            for sent in caption_item.sentences:
                tmp_tokens = sent.words + [Vocabulary.end_token_id]
                tmp_tokens = [_ if _ in vocab.word2idx else Vocabulary.unk_token for _ in tmp_tokens]
                ref_words.append(' '.join(tmp_tokens))
            refs_words.append(ref_words)
            count_imgs += 1

    print('total imgs:', count_imgs)

    ngram_words = compute_doc_freq(create_crefs(refs_words))
    return {'document_frequency': ngram_words, 'ref_len': count_imgs}

