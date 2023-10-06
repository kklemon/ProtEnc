from dataclasses import dataclass
import torch
import torch.nn as nn

from collections import OrderedDict
from typing import Callable
from enum import Enum
from transformers import (
    BertModel,
    BertTokenizer,
    T5EncoderModel,
    T5Tokenizer
)
from tensordict import TensorDict


class EmbeddingType(Enum):
    PER_RESIDUE = 'per_residue'
    PER_PROTEIN = 'per_protein'


class BaseProteinEmbeddingModel(nn.Module):
    embedding_type: EmbeddingType

    def prepare_sequences(self, sequences):
        return NotImplementedError

    def forward(self, input):
        raise NotImplementedError


def load_huggingface_language_model(model_cls, tokenizer_cls, model_name, load_weights=True):
    if load_weights:
        model = model_cls.from_pretrained(model_name)
        tokenizer = tokenizer_cls.from_pretrained(model_name)
        return model, tokenizer
    else:
        config = model_cls.config_class.from_pretrained(model_name)
        model = model_cls(config)
        tokenizer = tokenizer_cls.from_pretrained(model_name)
        return model, tokenizer


class BaseProtTransEmbeddingModel(BaseProteinEmbeddingModel):
    embedding_kind = EmbeddingType.PER_RESIDUE
    available_models = None

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer

    def _validate_model_name(self, model_name):
        assert self.available_models is None or model_name in self.available_models, \
            f'Unknown model name \'{model_name}\'. Available options are {self.available_models}'

    def prepare_sequences(self, sequences):
        # ProtTrans tokenizers expect whitespaces between residues
        sequences = [' '.join(s.replace(' ', '')) for s in sequences]

        return TensorDict(self.tokenizer.batch_encode_plus(
            sequences,
            return_tensors='pt',
            add_special_tokens=True,
            padding=True
        ), batch_size=len(sequences))

    def _post_process_embedding(self, embed, seq_len):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self,  input):
        attn_mask = input['attention_mask']

        output = self.model(**input)

        embeddings = output.last_hidden_state.cpu()
        seq_lens = (attn_mask == 1).sum(-1)

        for embed, seq_len in zip(embeddings, seq_lens):
            # Tokenized sequences have the following form:
            # [CLS] V N ... I K [SEP] [PAD] ... [PAD]
            #
            # We remove the special tokens ([CLS], [SEP], [PAD]) before
            # computing the mean over the remaining sequence

            yield self._post_process_embedding(embed, seq_len)


class ProtBERTEmbeddingModel(BaseProtTransEmbeddingModel):
    available_models = [
        'prot_bert',
        'prot_bert_bfd'
    ]

    def __init__(self, model_name: str, load_weights: bool = True):
        self._validate_model_name(model_name)

        mode_name = f'Rostlab/{model_name}'
        model, tokenizer = load_huggingface_language_model(
            BertModel,
            BertTokenizer,
            mode_name,
            load_weights=load_weights
        )

        super().__init__(model=model, tokenizer=tokenizer)

    def _post_process_embedding(self, embed, seq_len):
        return embed[1:seq_len - 1]


class ProtT5EmbeddingModel(BaseProtTransEmbeddingModel):
    available_models = [
        'prot_t5_xl_uniref50',
        'prot_t5_xl_bfd',
        'prot_t5_xxl_uniref50',
        'prot_t5_xxl_bfd'
    ]

    def __init__(self, model_name: str, load_weights: bool = True):
        self._validate_model_name(model_name)

        mode_name = f'Rostlab/{model_name}'
        model, tokenizer = load_huggingface_language_model(
            T5EncoderModel,
            T5Tokenizer,
            mode_name,
            load_weights=load_weights
        )

        super().__init__(model=model, tokenizer=tokenizer)

    def _post_process_embedding(self, embedding, seq_len):
        return embedding[:seq_len - 1]


class ESMEmbeddingModel(BaseProteinEmbeddingModel):
    embedding_kind = EmbeddingType.PER_RESIDUE

    def __init__(self, model_name: str, repr_layer: int):
        super().__init__()

        self.model, self.alphabet = torch.hub.load('facebookresearch/esm:main', model_name)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.repr_layer = repr_layer

    def prepare_sequences(self, sequences):
        _, _, batch_tokens = self.batch_converter([('', seq) for seq in sequences])

        return TensorDict({
            'tokens': batch_tokens
        }, batch_size=len(sequences))

    @torch.no_grad()
    def forward(self, input):
        results = self.model(**input, repr_layers=[self.repr_layer], return_contacts=True)
        token_representations = results['representations'][self.repr_layer]
        seq_lengths = (input['tokens'] != self.alphabet.padding_idx).sum(1)

        for i, seq_len in enumerate(seq_lengths):
            yield token_representations[i, 1:seq_len - 1]


@dataclass
class ModelCard:
    name: str
    family: str
    embed_dim: int
    init_fn: Callable[[], BaseProteinEmbeddingModel]

    @classmethod
    def from_model_cls(cls, *, model_cls, model_kwargs, **kwargs):
        def init_fn(**init_kwargs):
            return model_cls(**{**model_kwargs, **init_kwargs})

        return cls(init_fn=init_fn, **kwargs)


model_descriptions = [
    # ProtTrans family (https://github.com/agemagician/ProtTrans)
    ModelCard.from_model_cls(
        name='prot_t5_xl_uniref50',
        family='ProtTrans',
        embed_dim=1024,
        model_cls=ProtT5EmbeddingModel,
        model_kwargs=dict(model_name='prot_t5_xl_uniref50')
    ),
    ModelCard.from_model_cls(
        name='prot_t5_xl_bfd',
        family='ProtTrans',
        embed_dim=1024,
        model_cls=ProtT5EmbeddingModel,
        model_kwargs=dict(model_name='prot_t5_xl_bfd')
    ),
    ModelCard.from_model_cls(
        name='prot_t5_xxl_uniref50',
        family='ProtTrans',
        embed_dim=1024,
        model_cls=ProtT5EmbeddingModel,
        model_kwargs=dict(model_name='prot_t5_xxl_uniref50')
    ),
    ModelCard.from_model_cls(
        name='prot_t5_xxl_bfd',
        family='ProtTrans',
        embed_dim=1024,
        model_cls=ProtT5EmbeddingModel,
        model_kwargs=dict(model_name='prot_t5_xxl_bfd')
    ),
    ModelCard.from_model_cls(
        name='prot_bert_bfd',
        family='ProtTrans',
        embed_dim=1024,
        model_cls=ProtBERTEmbeddingModel,
        model_kwargs=dict(model_name='prot_bert_bfd')
    ),
    ModelCard.from_model_cls(
        name='prot_bert',
        family='ProtTrans',
        embed_dim=1024,
        model_cls=ProtBERTEmbeddingModel,
        model_kwargs=dict(model_name='prot_bert')
    ),

    # ESM family (https://github.com/facebookresearch/esm)
    ModelCard.from_model_cls(
        name='esm2_t48_15B_UR50D',
        family='ESM',
        embed_dim=5120,
        model_cls=ESMEmbeddingModel,
        model_kwargs=dict(model_name='esm2_t48_15B_UR50D', repr_layer=48)
    ),
    ModelCard.from_model_cls(
        name='esm2_t36_3B_UR50D',
        family='ESM',
        embed_dim=2560,
        model_cls=ESMEmbeddingModel,
        model_kwargs=dict(model_name='esm2_t36_3B_UR50D', repr_layer=36)
    ),
    ModelCard.from_model_cls(
        name='esm2_t33_650M_UR50D',
        family='ESM',
        embed_dim=1280,
        model_cls=ESMEmbeddingModel,
        model_kwargs=dict(model_name='esm2_t33_650M_UR50D', repr_layer=33)
    ),
    ModelCard.from_model_cls(
        name='esm2_t30_150M_UR50D',
        family='ESM',
        embed_dim=640,
        model_cls=ESMEmbeddingModel,
        model_kwargs=dict(model_name='esm2_t30_150M_UR50D', repr_layer=30)
    ),
    ModelCard.from_model_cls(
        name='esm2_t12_35M_UR50D',
        family='ESM',
        embed_dim=480,
        model_cls=ESMEmbeddingModel,
        model_kwargs=dict(model_name='esm2_t12_35M_UR50D', repr_layer=12)
    ),
    ModelCard.from_model_cls(
        name='esm2_t6_8M_UR50D',
        family='ESM',
        embed_dim=320,
        model_cls=ESMEmbeddingModel,
        model_kwargs=dict(model_name='esm2_t6_8M_UR50D', repr_layer=6)
    ),
]


model_dict: dict[str, ModelCard] = OrderedDict(
    (m.name, m) for m in model_descriptions
)

model_families = set(m.family for m in model_descriptions)


def list_models(family: str | None = None):
    if family is not None:
        if family not in model_families:
            raise ValueError(f'Unknown model family \'{family}\'. Available families are {model_families}')
        
        return [m.name for m in model_descriptions if m.family == family]
    else:
        return list(model_dict)
    

def get_model_info(model_name: str):
    if model_name not in model_dict:
        raise ValueError(f'Unknown model \'{model_name}\'. Available models are {list_models()}')

    model_desc = model_dict[model_name]

    return {
        'name': model_desc.name,
        'family': model_desc.family,
        'embed_dim': model_desc.embed_dim
    }


def get_model(model_name, **kwargs):
    model = model_dict[model_name].init_fn(**kwargs)
    return model
