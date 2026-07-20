from argparse import ArgumentParser
from Bio.PDB import PDBParser
from esm.utils.structure.protein_chain import ProteinChain
import torch
from multiprocessing import Pool
import os
import numpy as np
from tqdm import tqdm

from Bio import PDB
import numpy as np
import pandas as pd

aa_3to1 = {
    'ALA': 'A',
    'CYS': 'C',
    'ASP': 'D',
    'GLU': 'E',
    'PHE': 'F',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LYS': 'K',
    'LEU': 'L',
    'MET': 'M',
    'ASN': 'N',
    'PRO': 'P',
    'GLN': 'Q',
    'ARG': 'R',
    'SER': 'S',
    'THR': 'T',
    'VAL': 'V',
    'TRP': 'W',
    'TYR': 'Y'
}

def find_residue_triplet(pdb_file):
    # Initialize PDB parser
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    # Extract Cα coordinates and sequence positions
    ca_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue):  # Check if the residue is an amino acid
                    try:
                        ca = residue['CA']  # Get Cα atom
                        ca_atoms.append((residue.get_id()[1], ca.coord))  # (sequence_position, coordinates)
                    except KeyError:
                        continue  # Skip residues without Cα

    # Convert to numpy array for distance calculations
    positions = np.array([coords for _, coords in ca_atoms])
    sequence_positions = np.array([seq_pos for seq_pos, _ in ca_atoms])
    distance_matrix = np.linalg.norm(positions[:, None] - positions, axis=-1)

    # Find close residues
    close_residue_groups = []
    num_residues = len(ca_atoms)

    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            # Calculate distance between Cα atoms
            distance = distance_matrix[i, j]
            if distance < 6.0:  # Check if distance is less than 6 Å
                for k in range(j + 1, num_residues):
                    distance_kj = distance_matrix[k, j]
                    distance_jk = distance_matrix[j, k]
                    if distance_kj < 6.0 and distance_jk < 6.0:
                        # Check sequence distance
                        seq_i = sequence_positions[i]
                        seq_j = sequence_positions[j]
                        seq_k = sequence_positions[k]
                        if abs(seq_i - seq_j) >= 10 and abs(seq_i - seq_k) >= 10 and abs(seq_j - seq_k) >= 10:
                            # Add the group to results
                            close_residue_groups.append((seq_i, seq_j, seq_k))

    return np.array(close_residue_groups)


def process_pdb(pdb_path, out_path, is_predicted, df):

    if not os.path.exists(pdb_path):
        print(f'{pdb_path} does not exist')
        return
    if os.path.exists(out_path):
        print(f'{out_path} already exists')
        return

    try: 
        chain = ProteinChain.from_pdb(
            path=pdb_path, 
            chain_id='detect', 
            is_predicted=is_predicted
        )
        dict_ = {
            'sequence': chain.sequence,
            'atom37_positions': np.array(chain.atom37_positions),
            'confidence': np.array(chain.confidence)
        }

        triplets = find_residue_triplet(pdb_path)
        pdb_name = os.path.basename(pdb_path).replace('.pdb', '')
        df = df[df['Uniprot'] == pdb_name]
        assert len(df) == 1
        Knotcore_range = df['Knotcore_range'].values[0].split('-')
        Knotcore_range = [int(Knotcore_range[0]), int(Knotcore_range[1])]
        filtered_triplets = []
        for triplet in triplets:
            for t in triplet:
                if t >= Knotcore_range[0] and t <= Knotcore_range[1]:
                    filtered_triplets.append(triplet)
                    break
        dict_['triplets'] = np.array(filtered_triplets)
        np.save(out_path, dict_)
    except:
        print(f'Error in {pdb_path}')
        return


def main(args):

    df = pd.read_csv('/scratch/PI/hanyugao/puqing/esmknot/data/alphaknot_knotprob0.7_plddt70_length1024_nosimilar.csv')
    uniprot_ids = df['Uniprot'].values
    uniprot_ids = [id_ for id_ in uniprot_ids if not os.path.exists(os.path.join(args.out_dir, f'{id_}.npy'))]
    print(len(uniprot_ids))
    pdb_paths = [os.path.join(args.pdb_dir, f'{id_}.pdb') for id_ in uniprot_ids]
    out_paths = [os.path.join(args.out_dir, f'{id_}.npy') for id_ in uniprot_ids]
    is_predicted = [args.is_predicted] * len(uniprot_ids)
    dfs = [df] * len(uniprot_ids)

    with Pool(args.cpu) as p:
        list(p.starmap(process_pdb, zip(pdb_paths, out_paths, is_predicted, dfs)))


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--pdb_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--is_predicted', action='store_true')
    parser.add_argument('--cpu', type=int, default=8)
    args = parser.parse_args()

    main(args)
