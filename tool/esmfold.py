import os
import logging
import subprocess
import sys
from Bio.PDB import PDBParser
from typing import List, Tuple
import torch
import esm
from omegaconf import DictConfig
import typing as T
from tqdm import tqdm
import argparse
from Bio import SeqIO




def create_batched_sequence_datasest(
    sequences: T.List[T.Tuple[str, str]], max_tokens_per_batch: int = 1024
) -> T.Generator[T.Tuple[T.List[str], T.List[str]], None, None]:

    batch_headers, batch_sequences, num_tokens = [], [], 0
    for header, seq in sequences:
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)

    yield batch_headers, batch_sequences


def run_esmfold(seqs: T.List[T.Tuple[str, str]], pdb_dir: str):
    '''
    Monomer sequence prediction only
    '''
    seqs = sorted(seqs, key=lambda x: len(x[1]))
    seqs_to_pred = [(header, seq) for header, seq in seqs if not os.path.exists(os.path.join(pdb_dir, f"{header}.pdb"))]
    logging.info(f"Predicting the structure for {len(seqs_to_pred)} sequences")

    if len(seqs_to_pred) > 0:
        model = esm.pretrained.esmfold_v1()
        model = model.eval().cuda()
        model.set_chunk_size(128)        
        batched_sequences = list(create_batched_sequence_datasest(seqs_to_pred, max_tokens_per_batch=1280))
        for headers, sequences in tqdm(batched_sequences, desc="Predicting structures with ESMFold"):
            output = model.infer(sequences)
            output = {key: value.cpu() for key, value in output.items()}
            pdbs = model.output_to_pdb(output)
            for header, pdb_string in zip(headers, pdbs):
                path = os.path.join(pdb_dir, f"{header}.pdb")
                with open(path, "w") as f:
                    f.write(pdb_string)
        del model
        torch.cuda.empty_cache()  


if __name__ == '__main__':
    
    # argument parsing
    parser = argparse.ArgumentParser(description='Predict protein structure using ESMFold')
    parser.add_argument('--pmpnn_seq_file', type=str, help='File containing protein sequences')
    parser.add_argument('--pdb_dir', type=str, help='Directory to store pdb files')
    args = parser.parse_args()
    
    fasta_path = args.pmpnn_seq_file
    esmf_dir = args.pdb_dir

    seqs = []
    pdb_name = os.path.basename(fasta_path).replace('.fa', '')
    records = list(SeqIO.parse(fasta_path, "fasta"))
    for record in records:
        if 'sample=' in record.description:
            num = int(record.description.split('sample=')[1].split(',')[0])
            seqs.append((f'{pdb_name}_seq{num}', str(record.seq)))
    
    run_esmfold(seqs, esmf_dir)