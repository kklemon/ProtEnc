import argparse
import json
import pickle
import numpy as np
import lmdb

from Bio import SeqIO
from csv import DictReader
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Callable
from protenc.utils import HumanFriendlyParsingAction


class BaseInputReader(ABC):
    @staticmethod
    def add_arguments_to_parser(parser):
        pass

    @classmethod
    @abstractmethod
    def from_args(cls, path, args):
        pass

    @abstractmethod
    def __iter__(self):
        pass



class BaseOutputWriter:
    @staticmethod
    def add_arguments_to_parser(parser):
        pass

    @classmethod
    @abstractmethod
    def from_args(cls, path, args):
        pass

    @abstractmethod
    def __enter__(self) -> Callable[[str, np.ndarray], None]:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError


class CSVReader(BaseInputReader):
    def __init__(self,
                 path,
                 delimiter=',',
                 label_col='label',
                 sequence_col='protein'):
        self.path = Path(path)

        self.delimiter = delimiter
        self.label_col = label_col
        self.sequence_col = sequence_col

    @staticmethod
    def add_arguments_to_parser(parser: argparse.ArgumentParser):
        parser.add_argument('--csv_reader.delimiter', default=',')
        parser.add_argument('--csv_reader.label_col', default='label')
        parser.add_argument('--csv_reader.sequence_col', default='protein')

    @classmethod
    def from_args(cls, path, args):
        return cls(
            path,
            delimiter=args.csv_reader.delimiter,
            label_col=args.csv_reader.label_col,
            sequence_col=args.csv_reader.sequence_col
        )

    def __iter__(self):
        with self.path.open() as fp:
            reader = DictReader(fp, delimiter=self.delimiter)
            for row in reader:
                yield row[self.label_col], row[self.sequence_col]



class JSONReader(BaseInputReader):
    def __init__(self, path, stream: bool = False):
        self.path = Path(path)
        self.stream = stream

    @staticmethod
    def add_arguments_to_parser(parser: argparse.ArgumentParser):
        parser.add_argument('--json_reader.stream', action='store_true')

    @classmethod
    def from_args(cls, path, args):
        return cls(path, args.json_reader.stream)

    def __iter__(self):
        if self.stream:
            try:
                import json_stream
            except ImportError:
                raise ImportError('json_stream needs to be installed for streaming json input.')

            json_load = json_stream.load
        else:
            json_load = json.load

        with self.path.open() as fp:
            for label, protein in json_load(fp):
                yield label, protein


class FASTAReader(BaseInputReader):
    def __init__(self, path):
        self.path = Path(path)

    @classmethod
    def from_args(cls, path, args):
        return cls(path)

    def __iter__(self):
        with self.path.open() as fp:
            fasta_sequences = SeqIO.parse(fp, 'fasta')
            for fasta in fasta_sequences:
                label, sequence = fasta.id, str(fasta.seq)
                yield label, sequence



class LMDBWriter(BaseOutputWriter):
    def __init__(self, path, **lmdb_kwargs):
        self.path = path
        self.lmdb_kwargs = lmdb_kwargs
        self.ctx = None

    @staticmethod
    def add_arguments_to_parser(parser: argparse.ArgumentParser):
        parser.add_argument('--lmdb_writer.map_size', action=HumanFriendlyParsingAction, default=2 ** 30)
        parser.add_argument('--lmdb_writer.no_sub_dir', action='store_true')

    @classmethod
    def from_args(cls, path, args):
        return cls(
            path,
            map_size=args.lmdb_writer.map_size,
            subdir=not args.lmdb_writer.no_sub_dir
        )

    def _callback(self, label, embedding):
        assert self.ctx is not None
        assert isinstance(label, str) and isinstance(embedding, np.ndarray)

        _, txn = self.ctx
        txn.put(label.encode(), pickle.dumps(embedding))

    def __enter__(self) -> Callable[[str, np.ndarray], None]:
        if self.ctx is None:
            env = lmdb.open(str(self.path), **self.lmdb_kwargs)
            txn = env.begin(write=True)
            self.ctx = (env, txn)

        return self._callback

    def __exit__(self, exc_type, exc_val, exc_tb):
        env, txn = self.ctx
        txn.close()
        env.close()


input_format_mapping = {
    'csv': CSVReader,
    'json': JSONReader,
    'fasta': FASTAReader
}

output_format_mapping = {
    'lmdb': LMDBWriter
}
