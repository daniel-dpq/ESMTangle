from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data
from topoly import homfly
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import os
import subprocess
from Bio import SeqIO
import typing as T
import torch
from tqdm import tqdm
import esm
import logging
from Bio.PDB import PDBParser


import subprocess
import sys



def get_bb_positions(pdb_path):
    """
    Get backbone positions from a PDB file
    """
    s = get_structure(pdb_path)
    chains = list(s.get_chains())
    assert len(chains) == 1
    chain = chains[0]
    coords, _ = get_residue_data(chain)

    return coords


def align_structure(pdb1, pdb2):
    """
    Align two structures using TMalign
    """
    # Load structures
    if isinstance(pdb1, str):
        xyz1 = get_bb_positions(pdb1)
    else:
        xyz1 = pdb1
        assert len(xyz1.shape) == 2 and xyz1.shape[1] == 3

    if isinstance(pdb2, str):
        xyz2 = get_bb_positions(pdb2)
    else:
        xyz2 = pdb2
        assert len(xyz2.shape) == 2 and xyz2.shape[1] == 3
    
    assert xyz1.shape[0] == xyz2.shape[0], f"Number of residues do not match: {xyz1.shape[0]} vs {xyz2.shape[0]}\n{pdb1}\n{pdb2}"
    # Align
    res = tm_align(xyz1, xyz2, 'G' * len(xyz1), 'G' * len(xyz2))

    return {
        'rmsd': res.rmsd,
        'tm_norm_chain1': res.tm_norm_chain1,
        'tm_norm_chain2': res.tm_norm_chain2,
        'length': xyz1.shape[0],
    }


def detect_knot(pdb_path):

    if not os.path.exists(pdb_path):
        return 0, ''
        
    open_chain = homfly(pdb_path, tries=200)
    knot_prob = 0
    knot_type = ''
    for k, v in open_chain.items():
        if v > knot_prob:
            knot_prob = v
            knot_type = k
    
    return knot_prob, knot_type


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


def run_esmfold(seqs: T.List[T.Tuple[str, str]], pdb_dir: str, cfg: DictConfig):
    '''
    Monomer sequence prediction only
    '''
    activate_conda_env('esmfold_new')
    seqs = sorted(seqs, key=lambda x: len(x[1]))
    seqs_to_pred = [(header, seq) for header, seq in seqs if not os.path.exists(os.path.join(pdb_dir, f"{header}.pdb"))]
    logging.info(f"Predicting the structure for {len(seqs_to_pred)} sequences")

    if len(seqs_to_pred) > 0:
        model = esm.pretrained.esmfold_v1()
        model = model.eval().cuda()
        model.set_chunk_size(cfg.esmfold.chunk_size)        
        batched_sequences = list(create_batched_sequence_datasest(seqs_to_pred, max_tokens_per_batch=cfg.esmfold.max_tokens_per_batch))
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
    activate_conda_env('esm3')


def run_self_consistency(
        base_dir: str,
        cfg: DictConfig,
    ):
    """Run self-consistency on design proteins against reference protein.
    
    Args:
        base_dir: directory where files are stored. designs are stored in base_dir/design
        reference_pdb_path: path to reference protein file
        motif_mask: Optional mask of which residues are the motif.

    Returns:
        Writes ProteinMPNN outputs to base_dir/seqs
        Writes ESMFold outputs to base_dir/ef_structure
        Writes results in base_dir/sc_results.csv
    """
    csv_path = os.path.join(base_dir, 'sc_results.csv')
    if os.path.exists(csv_path):
        return       
    design_pdb_dir = os.path.join(base_dir, 'design')
    pdb_names = [f for f in os.listdir(design_pdb_dir) if f.endswith('.pdb')]
    mpnn_fasta_paths = [
        os.path.join(base_dir,'seqs', pdb_name.replace('.pdb', '.fa'))
        for pdb_name in pdb_names       
    ]

    # Run PorteinMPNN
    for mpnn_fasta_path in tqdm(mpnn_fasta_paths, desc="Running ProteinMPNN"):
        if os.path.exists(mpnn_fasta_path):
            continue
        pdb_name = os.path.basename(mpnn_fasta_path).replace('.fa', '')
        desin_pdb_path = os.path.join(design_pdb_dir, pdb_name + '.pdb') 
        num_tries = 0
        ret = -1
        pmpnn_args = [
            'python',
            f'{cfg.pmpnn.dir}/protein_mpnn_run.py',
            '--out_folder',
            base_dir,
            '--pdb_path',
            desin_pdb_path,
            '--num_seq_per_target',
            f'{cfg.pmpnn.num_seq_per_target}',
            '--sampling_temp',
            '0.1',
            '--seed',
            '38',
            '--batch_size',
            '1',
            '--device',
            '1',
        ]
        while ret < 0:
            try:
                process = subprocess.Popen(
                    pmpnn_args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT
                )
                ret = process.wait()
            except Exception as e:
                num_tries += 1
                logging.info(f'Failed ProteinMPNN. Attempt {num_tries}/5')
                torch.cuda.empty_cache()
                if num_tries > 4:
                    raise e

    # # Run ESMFold on each ProteinMPNN sequence
    # esmf_dir = os.path.join(base_dir, 'ef_structure')
    # os.makedirs(esmf_dir, exist_ok=True)
    # seqs = []
    # for fasta_path in mpnn_fasta_paths:
    #     pdb_name = os.path.basename(fasta_path).replace('.fa', '')
    #     records = list(SeqIO.parse(fasta_path, "fasta"))
    #     for record in records:
    #         if 'sample=' in record.description:
    #             num = int(record.description.split('sample=')[1].split(',')[0])
    #             seqs.append((f'{pdb_name}_seq{num}', str(record.seq)))
    
    # run_esmfold(seqs, esmf_dir, cfg)


def get_mean_plddt(pdb_path):
    # use biopython to get b factor
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('X', pdb_path)
    model = structure[0]
    plddt = []
    for chain in model:
        for residue in chain:
            for atom in residue:
                plddt.append(atom.get_bfactor())
    return np.mean(plddt)


def compute_self_consistency(design_dir, ef_dir, pmpnn_num_seq_per_target):

    results = []
    design_names = [f.replace('.pdb', '') for f in os.listdir(design_dir) if f.endswith('.pdb')]
    for design_name in tqdm(design_names, desc="Comparing generated structures and ESMFold-predicted structures"):
        design_path = os.path.join(design_dir, f'{design_name}.pdb')
        ef_paths = [os.path.join(ef_dir, f'{design_name}_seq{i}.pdb') for i in range(1, pmpnn_num_seq_per_target+1)]
        knot_prob, knot_type = detect_knot(design_path)
        for ef_path in ef_paths:
            res = align_structure(design_path, ef_path)
            res['design_name'] = os.path.basename(ef_path).replace('.pdb', '')
            res['knot_prob'] = knot_prob
            res['knot_type'] = knot_type
            results.append(res)
    
    return pd.DataFrame(results)