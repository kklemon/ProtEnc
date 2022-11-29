import argparse
import hashlib
import pickle
import random
import re

import lmdb
import torch

from itertools import chain
from collections.abc import Mapping
from humanfriendly import parse_size

# See https://www.drive5.com/usearch/manual/IUPAC_codes.html
from torch.utils.data import IterableDataset

nucleotide_wildcard_mapping = {
    'R': ['A', 'G'],
    'Y': ['C', 'T'],
    'S': ['G', 'C'],
    'W': ['A', 'T'],
    'K': ['G', 'T'],
    'M': ['A', 'C'],
    'B': ['C', 'G', 'T'],
    'D': ['A', 'G', 'T'],
    'H': ['A', 'C', 'T'],
    'V': ['A', 'C', 'G']
}
nucleotide_wildcards = set(nucleotide_wildcard_mapping)
nucleotide_wildcard_re = re.compile('(' + '|'.join(nucleotide_wildcards) + ')')


def identity(x, *args, **kwargs):
    return x


def merge_dicts(dicts, cat_fn=None):
    """
    Args:
        dicts: Dictionaries to merge key-wise.
        cat_fn (optional): Function to use for merging dictionary values.

    Example:
    >>> merge_dicts({'a': 0, 'b': 42}, {'a': 1})
    {'a': [0, 1], 'b': [42]}
    """
    if cat_fn is None:
        cat_fn = identity

    keys = set(chain.from_iterable(dicts))
    merged = {k: cat_fn([d[k] for d in dicts if k in d]) for k in keys}
    return merged


def digest_strings(ss):
    m = hashlib.md5()
    for s in ss:
        m.update(s.encode())
    return m.hexdigest()


def get_lmdb_keys(env):
    with env.begin() as txn:
        return list(txn.cursor().iternext(values=False))


def to_device(o, device=None):
    if isinstance(o, torch.Tensor):
        return o.to(device)
    if isinstance(o, (list, tuple)):
        return type(o)(to_device(el, device) for el in o)
    if isinstance(o, Mapping):
        return {k: to_device(v, device) for k, v in o.items()}
    return o


def sub_nucleotide_wildcards(seq):
    res = seq
    for match in nucleotide_wildcard_re.finditer(seq):
        res = res[:match.start()] + random.choice(nucleotide_wildcard_mapping[match.group()]) + res[match.end():]
    return res


def read_from_lmdb(path):
    with lmdb.open(str(path), readonly=True) as env:
        with env.begin() as txn, txn.cursor() as cursor:
            for key, value in cursor.iternext():
                yield key.decode(), pickle.loads(value)


class IteratorWrapper(IterableDataset):
    def __init__(self, it):
        self.it = it

    def __iter__(self):
        yield from self.it


class HumanFriendlyParsingAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, list):
            values = [parse_size(v) for v in values]
        else:
            values = parse_size(values)

        setattr(namespace, self.dest, values)


class NestedNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setattr__(self, key, value):
        parent, *children = key.split('.', maxsplit=1)
        if children:
            existing = {}
            if hasattr(self, parent):
                existing = vars(getattr(self, parent))
            super().__setattr__(parent, NestedNamespace(**{**existing, children[0]: value}))
        else:
            super().__setattr__(key, value)
