import inspect
import time
from collections.abc import Iterable

from .customjson import *
from .data import *
from .vocabulary import *
from .pipeline import *
from .loss import *
from .evaluate import *


# force check parameter type
def arg_type(**decorator_kwargs):
    def real_decorator(func):
        real_func_arg_names = inspect.getfullargspec(func).args
        arg_idx_to_arg_name = dict(enumerate(real_func_arg_names))
        args_to_check = decorator_kwargs
        # print('real args:', real_func_arg_names)
        # print('args to check:', args_to_check)
        _args_to_check = {}
        for k, v in args_to_check.items():
            assert k in real_func_arg_names
            if isinstance(v, Iterable):     # a collection of types
                for t in v:
                    assert isinstance(t, type), 'specified invalid type {} for argument {}'.format(t, k)
                _args_to_check[k] = v
            else:
                assert isinstance(v, type), 'specified invalid type {} for argument {}'.format(v, k)
                _args_to_check[k] = [v]
        args_to_check = _args_to_check

        def wrapper(*args, **kwargs):
            start_time = time.time()
            for i, arg_value in enumerate(args):
                arg_name = arg_idx_to_arg_name[i]
                if arg_name not in args_to_check:
                    continue
                assert type(arg_value) in args_to_check[arg_name], \
                    'argument \'{}\' expected type {}, but received type {}'\
                        .format(arg_name, args_to_check[arg_name], type(arg_value))
            for arg_name, arg_value in kwargs.items():
                if arg_name not in args_to_check:
                    continue
                assert type(arg_value) in args_to_check[arg_name], \
                    'argument \'{}\' expected type {}, but received type {}'\
                        .format(arg_name, args_to_check[arg_name], type(arg_value))
            # print('check args used {}s'.format(time.time() - start_time))
            return func(*args, **kwargs)
        return wrapper
    return real_decorator


def get_word_mask(shape, sent_lengths):
    assert shape[0] == len(sent_lengths)
    mask = np.zeros(shape, dtype=np.int64)
    for i in range(shape[0]):
        mask[i, :sent_lengths[i]] = 1
    return mask


@arg_type(tokens=[list, np.ndarray])
def trim_generated_tokens(tokens):
    i = 0
    while i < len(tokens) and tokens[i] == Vocabulary.start_token_id: i += 1
    start_index = i
    while i < len(tokens) and tokens[i] != Vocabulary.end_token_id: i += 1
    end_index = i
    return tokens[start_index : end_index]


