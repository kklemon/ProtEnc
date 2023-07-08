import torch
import protenc.utils as utils

from functools import cached_property
from protenc.types import BatchSize, ProteinEncoderInput, ReturnFormat
from torch.utils.data import DataLoader
from protenc.models import BaseProteinEmbeddingModel, get_model


class ProteinEncoder:
    def __init__(
        self,
        model: BaseProteinEmbeddingModel,
        batch_size: BatchSize = None,
        preprocess_workers: int = 0
    ):
        self.model = model
        self.batch_size = 1 if batch_size is None else batch_size
        self.preprocess_workers = preprocess_workers


    @cached_property
    def device(self):
        return next(iter(self.model.parameters())).device
    
    def _get_data_loader(self, proteins: list[str]):
        assert isinstance(self.batch_size, int), 'batch size must be provided as integer at the moment'
        return DataLoader(
            proteins,
            collate_fn=self.model.prepare_sequences,
            batch_size=self.batch_size,
            num_workers=self.preprocess_workers
        )
    
    def _encode(
        self,
        proteins: list[str],
        average_sequence: bool = False,
        return_format: ReturnFormat = 'torch'
    ):
        batches = self._get_data_loader(proteins)

        for batch in batches:
            batch = utils.to_device(batch, self.device)
            
            for embed in self.model(batch):
                if average_sequence:
                    embed = embed.mean(0)
                
                yield utils.to_return_format(embed.cpu(), return_format)

    def encode(
        self,
        proteins: ProteinEncoderInput,
        average_sequence: bool = False,
        autocast: bool = False,
        return_format: ReturnFormat = 'torch'
    ):
        with torch.autocast('cuda', enabled=autocast):
            if isinstance(proteins, dict):
                yield from zip(
                    proteins.keys(),
                    self.encode(
                        list(proteins.values()),
                        average_sequence=average_sequence,
                        return_format=return_format
                    )
                )
            elif isinstance(proteins, list):
                yield from self._encode(
                    proteins,
                    average_sequence=average_sequence,
                    return_format=return_format
                )
            else:
                raise TypeError('Expected list of proteins sequences or dictionary with protein '
                                f'sequences as values but found {type(proteins)}')
    
    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


def get_encoder(model_name, device=None, **kwargs):
    model = get_model(model_name, device)
    return ProteinEncoder(model, **kwargs)
