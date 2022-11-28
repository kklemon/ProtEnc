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

### Installation

```bash
pip install protenc
```

**Note:** while protenc depends on [pytorch](https://pytorch.org/), it is not part of the formal dependencies. 
This is due to the large number of different pytorch distributions which may mismatch with the target environment.

### Python API

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

# Use GPU if available
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

### Command-line interface

After installation, use the `protenc` shell command for bulk generation and export of protein embeddings.

```bash
protenc <path-to-protein-sequences> <path-to-output> --model_name=<name-of-model>
```

By default, input and output formats are inferred from the file extensions.

Run

```bash
protenc --help
```

for a detailed usage description.

**Example**

Generate protein embeddings using the ESM2 650M model for sequences provided in a [FASTA](https://en.wikipedia.org/wiki/FASTA_format) file and write embeddings to an [LMDB](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database):

```bash
protenc proteins.fasta embeddings.lmdb --model_name=esm2_t33_650M_UR50D
```

The generated embeddings will be stored in a lmdb key-value store and can be easily accessed using the `read_from_lmdb` utility function:

```python
from protenc.utils import read_from_lmdb

for label, embed in read_from_lmdb('embeddings.lmdb'):
    print(label, embed)
```

**Features**

Input formats:
* CSV
* JSON
* [FASTA](https://en.wikipedia.org/wiki/FASTA_format)

Output format:
* [LMDB](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database)
* [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) (coming soon)

General:
* Multi-GPU inference with (`--data_parallel`)
* FP16 inference (`--amp`)

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
  - [X] FASTA
  - [X] JSON
- [ ] Support for more output formats
  - [X] LMDB
  - [ ] HDF5
  - [ ] DataFrame
  - [ ] Pickle
- [ ] Large models support
  - [ ] Model offloading
  - [ ] Sharding
- [ ] Support for more protein language models
  - [X] Whole ProtTrans family
  - [X] Whole ESM family
  - [ ] AlphaFold (?)
- [X] Implement all remaining TODOs in code
- [ ] Distributed inference
- [ ] Maybe support some sort of optimized inference such as quantization
  - This may be up to the model providers
- [ ] Improve documentation
- [ ] Support translation of gene sequences
