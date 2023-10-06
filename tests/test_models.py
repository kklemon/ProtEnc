import pytest
import torch

from .fixtures import protein_dict, proteins
from protenc.models import get_model, get_model_info
from .utils import list_models_to_test, skip_no_gpu


@skip_no_gpu
@pytest.mark.parametrize('model_name', list_models_to_test())
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_protein_language_model(
    model_name,
    proteins,
    device
):
    model = get_model(model_name).to(device)
    model_info = get_model_info(model_name)

    batch = model.prepare_sequences(proteins).to(device)
    embeds = list(model(batch))
    
    for prot, embed in zip(proteins, embeds):
        assert len(prot) == len(embed)
        assert embed.shape[-1] == model_info['embed_dim']
