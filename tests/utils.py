from functools import wraps
import torch
import pytest
import warnings

from protenc.models import get_model_info, list_models


def skip_no_gpu(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        device = kwargs.get('device', 'cpu')

        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('No GPU available')
        
        return fn(*args, **kwargs)
    
    return wrapper


def skip_large_models(max_embed_dim=None):
    def wrap(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            model_name = kwargs.get('model_name')

            if model_name is None:
                model_info = get_model_info(model_name)
                if model_info['embed_dim'] > max_embed_dim:
                    pytest.skip(f'Model too large ({model_info["embed_dim"]} > {max_embed_dim} embed dimensions)')
            else:
                warnings.warn('Test decorated with @skip_large_model but no model_name argument found. '
                              'This is probably a mistake.')

            return fn(*args, **kwargs)
        
        return wrapper
    
    return wrap


def list_models_to_test(max_embed_dim=512):
    return [
        model_name
        for model_name in list_models()
        if get_model_info(model_name)['embed_dim'] <= max_embed_dim
    ]
