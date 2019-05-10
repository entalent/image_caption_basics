import inspect
import time

from .customjson import *
from .data import *
from .vocabulary import *
from .pipeline import *


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
            if isinstance(v, type):
                _args_to_check[k] = [v]
            else:
                _args_to_check[k] = v
        args_to_check = _args_to_check

        def wrapper(*args, **kwargs):
            start_time = time.time()
            for i, arg_value in enumerate(args):
                arg_name = arg_idx_to_arg_name[i]
                assert type(arg_value) in args_to_check[arg_name], \
                    'argument \'{}\' expected type {}, but received type {}'\
                        .format(arg_name, args_to_check[arg_name], type(arg_value))
            for arg_name, arg_value in kwargs.items():
                assert type(arg_value) in args_to_check[arg_name], \
                    'argument \'{}\' expected type {}, but received type {}'\
                        .format(arg_name, args_to_check[arg_name], type(arg_value))
            # print('check args used {}s'.format(time.time() - start_time))
            return func(*args, **kwargs)
        return wrapper
    return real_decorator