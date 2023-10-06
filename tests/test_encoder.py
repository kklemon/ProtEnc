import pytest
import torch

from protenc.models import get_model, get_model_info
from protenc.encoder import get_encoder
from .fixtures import protein_dict, proteins
from .utils import list_models_to_test, skip_no_gpu, skip_large_models


@pytest.mark.parametrize('model_name', list_models_to_test())
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@skip_no_gpu
@skip_large_models(max_embed_dim=1280)
def test_get_encoder(model_name, device):
    encoder = get_encoder(model_name, device=device)
    model = get_model(model_name, device=device)

    for encoder_param, model_param in zip(encoder.model.parameters(), model.parameters()):
        assert torch.allclose(encoder_param, model_param)
        assert encoder_param.device == device


@skip_no_gpu
@skip_large_models(max_embed_dim=1280)
@pytest.mark.parametrize('model_name', list_models_to_test())
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_encode(proteins, model_name, device):
    model_info = get_model_info(model_name)
    encoder = get_encoder(model_name, device=device)

    for prot, embed in zip(proteins, encoder(proteins)):
        assert len(prot) == len(embed)
        assert embed.shape[-1] == model_info['embed_dim']


@skip_no_gpu
@skip_large_models(max_embed_dim=1280)
@pytest.mark.parametrize('model_name', list_models_to_test())
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_encode_dict(protein_dict, model_name, device):
    model_info = get_model_info(model_name)
    encoder = get_encoder(model_name, device=device)

    for prot_id, embed in encoder(protein_dict):
        assert len(protein_dict[prot_id]) == len(embed)
        assert embed.shape[-1] == model_info['embed_dim']


@skip_no_gpu
@skip_large_models(max_embed_dim=1280)
@pytest.mark.parametrize('model_name', list_models_to_test())
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_encode_dict(protein_dict, model_name, device):
    model_info = get_model_info(model_name)
    encoder = get_encoder(model_name, device=device)

    for prot_id, embed in encoder(protein_dict):
        assert len(protein_dict[prot_id]) == len(embed)
        assert embed.shape[-1] == model_info['embed_dim']
