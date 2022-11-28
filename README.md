protenc
=======

protenc is a library to simplify extraction of protein embeddings from various pre-trained models, including:

* [ProtTrans](https://github.com/agemagician/ProtTrans) family
* [ESM](https://github.com/facebookresearch/esm)
* AlphaFold (coming soon™)

It provides a programmatic Python API as well as a highly flexible bulk extraction script, supporting many input and
output formats.

**Note:** the project is work in progress.

Usage
-----

**Installation**

```bash
pip install protenc
```

**Python API**

```python
import protenc
import torch

# List available models
print(protenc.list_models())

# Instantiate a model
model = protenc.get_model('esm2_t33_650M_UR50D')

proteins = [
  'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
  'KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE'
]

batch = model.prepare_sequences(proteins)

# Move to GPU if available
if torch.cuda.is_available():
  model = model.to('cuda')
  batch = protenc.utils.to_device(batch, 'cuda')

for embed in model(batch):
  # Embeddings have shape [L, D] where L is the sequence length and D the 
  # embedding dimensionality.
  print(embed.shape)
  
  # Derive a single per-protein embedding vector by averaging along the 
  # sequence dimension
  embed.mean(0)
```

**Command-line interface**

Coming soon.

Development
-----------

Clone the repository:

```bash
git clone git+https://github.com/kklemon/protenc.git
```

Install dependencies via [Poetry](https://python-poetry.org/):

```bash
poetry install
```

Todo
----

- [ ] Support for more input formats
  - [X] CSV
  - [ ] Parquet
  - [ ] FASTA
  - [ ] JSON
- [ ] Support for more output formats
  - [X] LMDB
  - [ ] HDF5
  - [ ] DataFrame
  - [ ] Pickle
- [ ] Large models support
  - [ ] Model offloading
  - [ ] Sharding
- [ ] Support for more protein language models
  - [ ] While ProtTrans family
  - [ ] While ESM family
    - [ ] AlphaFold (?)
- [ ] Implement all remaining TODOs in code
- [ ] Distributed inference
- [ ] Maybe support some sort of optimized inference such as quantization
  - This may be up to the model providers
- [ ] Improve documentation
- [ ] Support translation of gene sequences
