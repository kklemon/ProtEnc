import argparse
import os
import textwrap
import torch
import torch.nn as nn
import protenc

from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from protenc import io as io, utils
from protenc.models import EmbeddingKind


def get_input_reader_cls(args):
    input_path = Path(args.input_path)
    infer = args.input_format in [None, 'infer']
    if infer:
        input_format = input_path.suffix[1:]
    else:
        input_format = args.input_format

    cls = io.input_format_mapping.get(input_format)

    if cls is None:
        raise ValueError(f'Unknown input format \'{input_format}\'' + ' (inferred)' if infer else '')

    return cls


def get_output_reader_cls(args):
    output_path = Path(args.output_path)
    infer = args.input_format in [None, 'infer']
    if infer:
        output_format = output_path.suffix[1:]
    else:
        output_format = args.output_format

    cls = io.output_format_mapping.get(output_format)

    if cls is None:
        raise ValueError(f'Unknown input format \'{output_format}\'' + ' (inferred)' if infer else '')

    return cls


def collate_fn(samples, max_len=None, transform_fn=None):
    labels, sequences = zip(*samples)

    if max_len:
        sequences = [s[:max_len] for s in sequences]

    if transform_fn is not None:
        sequences = transform_fn(sequences)

    return labels, sequences


@torch.no_grad()
def main(args):
    model = protenc.get_model(args.model_name)
    model.eval()

    print(f'Reading data from {args.input_path}')

    input_reader = args.input_reader_cls.from_args(args.input_path, args)

    transform_fn = model.prepare_sequences
    if args.substitute_wildcards:
        transform_fn = lambda seqs: transform_fn([utils.sub_nucleotide_wildcards(s) for s in seqs])

    batches = DataLoader(
        utils.IteratorWrapper(input_reader),
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, max_len=args.max_prot_len, transform_fn=transform_fn),
        num_workers=args.num_workers
    )

    if 'cuda' in args.device:
        if args.data_parallel:
            model.model = nn.DataParallel(model.model, device_ids=args.device_ids or None)

    model = model.to(args.device)

    print(f'Starting inference loop; writing to {args.output_path}')

    with (
        torch.cuda.amp.autocast(enabled=args.amp),
        args.output_writer_cls.from_args(args.output_path, args) as writer,
        tqdm() as pbar
    ):
        for labels, sequences in batches:
            sequences = utils.to_device(sequences, args.device)
            output = model(sequences)

            for label, embedding in zip(labels, output):
                embedding = embedding.cpu().numpy()

                if args.compute_mean and model.embedding_kind == EmbeddingKind.PER_RESIDUE:
                    assert embedding.ndim == 2
                    embedding = embedding.mean(0)

                if args.cast_to:
                    embedding = embedding.astype(args.cast_to)

                if not args.dry_run:
                    # TODO: having the put call on the same thread/process as the GPU calls probably results in stalls.
                    #   This may be improved by setting lmdb.open(...) options appropriately or moving the storing
                    #   procedure into its own thread or process.
                    writer(label, embedding)

            pbar.update(len(labels))


def entrypoint():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent('''
            Script to bulk generate protein embeddings for given protein sequences, supporting various pre-trained
            protein language models, a variety of data formats and multi-GPU inference.
            ''')
    )
    parser.add_argument('input_path',
                        help='Path to file containing protein sequence. '
                             f'Currently supported formats are {list(io.input_format_mapping)}. '
                             f'File format is inferred from file extension if not set explicitly '
                             f'with the --input_format option.')
    parser.add_argument('output_path',
                        help='Path to output where embeddings should be written.  '
                             f'Currently supported formats are {list(io.output_format_mapping)}. '
                             f'File format is inferred from file extension if not set explicitly '
                             f'with the --output_format option.')
    parser.add_argument('--input_format', default='infer', choices=['infer', 'fasta', 'csv', 'json'],
                        help=f'Data format of input. Supported formats are {list(io.input_format_mapping)}. '
                             f'Will be inferred from input path by default.')
    parser.add_argument('--output_format', default='infer', choices=['infer', 'parquet', 'pickle', 'lmdb'],
                        help=f'Data format of output. Supported formats are {list(io.output_format_mapping)}. '
                             f'Will be inferred from output path by default.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size. Note, that a batch will be distributed to multiple GPUs if the '
                             '--data_parallel flag is set. The value should be adjusted accordingly.')
    parser.add_argument('--model_name', default='prot_bert', choices=protenc.list_models(),
                        help='Name / ID of the embedding model to be loaded.')
    parser.add_argument('--data_parallel', action='store_true',
                        help='Use multiple GPUs with torch.nn.DataParallel.')
    parser.add_argument('--amp', action='store_true',
                        help='Automatic mixed precision aka FP16. Can reduce memory consumption '
                             'and speed up inference if supported.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--device_ids', type=int, nargs='*',
                        help='Which devices to use for data parallelism.')
    parser.add_argument('--no_gpu', action='store_true',
                        help='Use the CPU for inference.')
    parser.add_argument('--compute_mean', '--pool', action='store_true',
                        help='Compute the average over the sequence axis of embeddings. '
                             'Applies only if the used model does produce sequence-wise outputs.')
    parser.add_argument('--substitute_wildcards', action='store_true',
                        help='Substitute amino acid wildcards by possible substitutes.')
    parser.add_argument('--cast_to',
                        help='Cast embedding arrays to a numpy data type.')
    parser.add_argument('--max_prot_len', type=int, default=512,
                        help='Maximum length of protein sequences. '
                             'Note: this should always be set to some value as length outliers '
                             'may produce high memory peaks for attention-based models.')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                        help='Number of parallel workers for preprocessing and batching. '
                             'Set to 0 for using the main process only. '
                             'Defaults to number of cpu cores on the machine.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Perform a dry run, i.e. do not write results.')

    args, rem_args = parser.parse_known_args(namespace=utils.NestedNamespace())

    args.input_reader_cls = get_input_reader_cls(args)
    args.input_reader_cls.add_arguments_to_parser(parser)

    args.output_writer_cls = get_output_reader_cls(args)
    args.output_writer_cls.add_arguments_to_parser(parser)

    parser.parse_args(namespace=args)

    main(args)
