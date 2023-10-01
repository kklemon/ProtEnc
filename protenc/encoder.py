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
        autocast: bool = False,
        preprocess_workers: int = 0
    ):
        self.model = model
        self.batch_size = 1 if batch_size is None else batch_size
        self.autocast = autocast
        self.preprocess_workers = preprocess_workers

    @cached_property
    def device(self):
        return next(iter(self.model.parameters())).device
    
    def _get_data_loader(self, proteins: list[str]):
        assert isinstance(self.batch_size, int), 'batch size must be provided as integer at the moment'
        return DataLoader(
            proteins,
            collate_fn=self.prepare_sequences,
            batch_size=self.batch_size,
            num_workers=self.preprocess_workers
        )
    
    def prepare_sequences(self, proteins: list[str]):
        return self.model.prepare_sequences(proteins)
    
    def _encode(self, batch):
        with (
            torch.inference_mode(),
            torch.autocast('cuda', enabled=self.autocast)
        ):
            return self.model(batch)
        
    def _encode_batches(
        self,
        proteins: list[str],
        average_sequence: bool = False,
        return_format: ReturnFormat = 'torch'
    ):
        batches = self._get_data_loader(proteins)

        for batch in batches:
            batch = batch.to(self.device)
            
            for embed in self._encode(batch):
                if average_sequence:
                    embed = embed.mean(0)
                
                yield utils.to_return_format(embed.cpu(), return_format)

    def encode(
        self,
        proteins: ProteinEncoderInput,
        average_sequence: bool = False,
        return_format: ReturnFormat = 'torch'
    ):
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
            yield from self._encode_batches(
                proteins,
                average_sequence=average_sequence,
                return_format=return_format
            )
        else:
            raise TypeError('Expected list of proteins sequences or dictionary with protein '
                            f'sequences as values but found {type(proteins)}')
    
    def encode_batch(
        self,
        proteins: list[str],
        average_sequence: bool = False,
        return_format: ReturnFormat = 'torch'
    ):
        batch = self.prepare_sequences(proteins)
        batch = batch.to(self.device)

        embeds = self._encode(batch)

        if average_sequence:
            embeds = embeds.mean(1)
        
        return utils.to_return_format(embeds.cpu(), return_format)
        
    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


def get_encoder(model_name, device=None, **kwargs):
    model = get_model(model_name, device)
    return ProteinEncoder(model, **kwargs)
