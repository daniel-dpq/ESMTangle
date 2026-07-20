import os
import numpy as np
import pandas as pd
from Bio import PDB
from tqdm import tqdm
from multiprocessing import Pool
from argparse import ArgumentParser
import torch
from typing import Dict, Tuple
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, ESMProteinTensor, GenerationConfig
from esm.utils.structure.protein_chain import ProteinChain
from peft import LoraConfig, TaskType, get_peft_model

from topoly import homfly
import math

from huggingface_hub import login
hf_token = "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"  

login_success = False
while login_success is False:
    try:
        login(token=hf_token)
        login_success = True
    except:
        pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# seed all
def seed_all(seed: int):
    """
    Set the seed for all random number generators.
    Args:
        seed: The seed to set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)



def sample(args):
    processed_data, triplet_prompt, out_dir, model = args

    result_path = os.path.join(out_dir, f"{processed_data['id']}.csv")
    if os.path.exists(result_path):
        print(f"File {result_path} already exists, skipping...")
        return

    def expand_triplet(triplet, L):
        """
        Expand triplet, determine whether to mask the left or right residue with 50% probability.
        Args:
            triplet: Close residue triplet, (3,) containing residue indices starting from 1
            L: Length of the sequence
        Returns:
            triplet: Expanded triplet, (3-9,)   indices starting from 0
        """
        triplet = torch.tensor(triplet, dtype=torch.int32)
        assert triplet[2] <= L, f' Triplet:{triplet} L:{L}'
        to_add = []
        for i in range(3):
            if torch.rand(1) < 0.5 and triplet[i]-1 > 0:
                to_add.append(triplet[i]-1)
            if torch.rand(1) < 0.5 and triplet[i]+1 <= L:
                to_add.append(triplet[i]+1)
        if len(to_add) > 0:
            triplet = torch.cat([triplet, torch.tensor(to_add, dtype=torch.int32)])
        assert 3 <= len(triplet) and len(triplet) <= 9, f' Triplet:{triplet} L:{L}'
        triplet = triplet - 1  # Convert to 0-based index
        return triplet.to(torch.int32)
    
    def sample_mask(L):
        'True means masked, False means unmasked'
        mask = torch.zeros(L, dtype=torch.bool)
        u = torch.rand(1)
        mask_rate = min(torch.cos(math.pi/2*u), (L-3)/L)
        mask_num = max(int(mask_rate * L), 4)
        if torch.rand(1) < 0.5:
            # Span masking
            num_unmasks = L - mask_num
            start_idx = torch.randint(0, L - num_unmasks + 1, (1,)).item()
            mask[start_idx:start_idx + num_unmasks] = True
            mask = ~ mask
        else:
            # Random masking
            # mask = torch.rand((L,)) < mask_rate  # [L,]
            mask_idx = torch.randperm(L)[:mask_num]
            mask[mask_idx] = True
        return mask

    length = len(processed_data['sequence'])
    sequence_prompt = ["_"] * length
    structure_prompt = torch.full((length, 37, 3), np.nan)

    if triplet_prompt:
        triplet = processed_data['triplets'][torch.randint(0, len(processed_data['triplets']), (1,)).item()]
        motif_idx = expand_triplet(triplet, length)
    else: 
        m = sample_mask(length)
        motif_idx = torch.arange(length)[~m]

    for idx in motif_idx:
        sequence_prompt[idx] = processed_data['sequence'][idx]
        structure_prompt[idx] = torch.from_numpy(processed_data['atom37_positions'][idx])
    sequence_prompt = "".join(sequence_prompt)

    protein_prompt = ESMProtein(sequence=sequence_prompt, coordinates=structure_prompt)
    template_chain = protein_prompt.to_protein_chain()

    generation_config = GenerationConfig(
        track="structure",  # We want ESM3 to generate tokens for the sequence track
        num_steps=sequence_prompt.count("_") // 4,  # We'll use num(mask tokens) // 2 steps to decode the sequence
        temperature=0.7,  # We'll use a temperature of 0.5 to control the randomness of the decoding process
    )

    ptms = []
    ids = []
    site_rmsds = []
    for i in range(16):
        id_ = processed_data['id'] + f"_design{i}"
        design_path = os.path.join(out_dir, f"{id_}.pdb")
        protein = model.generate(protein_prompt, generation_config)
        protein.sequence = protein.sequence.replace('_', 'G')
        protein.to_pdb(design_path)
        ptms.append(protein.ptm.item()) if protein.ptm is not None else ptms.append(None)
        ids.append(id_)
        # site RMSD
        chain = protein.to_protein_chain()
        site_rmsd = chain[motif_idx].rmsd(template_chain[motif_idx])
        site_rmsds.append(site_rmsd)
    
    # motif_idx to list
    motif_idx = motif_idx.tolist()
    motif_idx = [str(i) for i in motif_idx]
    motif_idx = ",".join(motif_idx)
    result_path = os.path.join(out_dir, f"{processed_data['id']}.csv")
    df = pd.DataFrame({'id': ids, 'ptm': ptms, 'site_rmsd': site_rmsds, 'motif_idx': [motif_idx] * len(ids)})
    df.to_csv(result_path, index=False)

    # template_chain = protein_prompt.to_protein_chain()
    # generation_chain = protein_design.to_protein_chain()

    # constrained_site_rmsd = template_chain[motif_idx].rmsd(
    #     generation_chain[motif_idx]
    # )
    # backbone_rmsd = template_chain.rmsd(generation_chain)
    # print(f"Constrained site RMSD: {constrained_site_rmsd:.2f} \t Backbone RMSD: {backbone_rmsd:.2f}")


def generate_design(args):

    seed_all(42)
    os.makedirs(args.out_dir, exist_ok=True)

    # df = pd.read_csv('/scratch/PI/hanyugao/puqing/esmknot/data/alphaknot_knotprob0.7_plddt70_length1024_nosimilar.csv')
    # df = df[df['pLDDT_chain'] >= 90]
    # df = df[df['Chain_length'] <= 512]
    # df = df[df['coil_percent'] <= 0.5]
    # df = df[df['Knot_probability'] >= 0.7]
    # df = df[df['Uniprot'].apply(lambda x: os.path.exists(os.path.join(args.processed_dir, f"{x}.npy")))]
    # print(f'Number of records: {len(df)}')
    # cluster = pd.read_csv('/scratch/PI/hanyugao/puqing/esmknot/data/alphaknot_knotprob0.7_plddt70_length1024_nosimilar_clust70.tsv', sep='\t', header=None, names=['rep_id', 'seq_id'])
    # cluster = cluster[cluster['seq_id'].isin(df['Uniprot'])]
    # cluster = cluster.groupby('rep_id').agg({'seq_id': lambda x: list(x)[0]}).reset_index()
    # uniprot_ids = cluster['seq_id'].unique().tolist()
    # print(f'Number of unique uniprot ids: {len(uniprot_ids)}')

    processed_files =[f for f in os.listdir(args.processed_dir) if f.endswith('.npy')]
    processed_ids = [f.replace('.npy', '') for f in processed_files]
    processed_paths = [
        os.path.join(args.processed_dir, f) for f in processed_files
    ]
    processed_data = [
        np.load(path, allow_pickle=True).item() for path in processed_paths
    ]

    for i in range(len(processed_data)):
        processed_data[i]['id'] = processed_ids[i]

    processed_data_with_triplet = [
        data for data in processed_data if len(data['triplets']) > 0
    ]
    processed_data_without_triplet = [
        data for data in processed_data if len(data['triplets']) == 0
    ]
    print(f'{len(processed_data_with_triplet)} records with triplet, {len(processed_data_without_triplet)} records without triplet')

    if len(processed_data_with_triplet) > len(processed_data) // 2: # Only 50% use triplet prompt
        processed_data_without_triplet += processed_data_with_triplet[len(processed_data) // 2:]
        processed_data_with_triplet = processed_data_with_triplet[:len(processed_data) // 2]
    
    processed_data = processed_data_with_triplet + processed_data_without_triplet
    triplet_prompts = [True] * len(processed_data_with_triplet) + [False] * len(processed_data_without_triplet)

    if args.range[1] > len(processed_data) or args.range[1] == -1:
        args.range[1] = len(processed_data)
    print(f'Range: {args.range[0]} - {args.range[1]}')
    processed_data = processed_data[args.range[0]:args.range[1]]
    triplet_prompts = triplet_prompts[args.range[0]:args.range[1]]

    model: ESM3InferenceClient = ESM3.from_pretrained('esm3-open')
    if args.lora_ckpt is not None:
        if '130000' in args.lora_ckpt:
            config = LoraConfig(
                target_modules=["out_proj",  "ffn.1", "fnn.3", 'structure_head.0', 'structure_head.3'],
                r=256,  
                lora_alpha=512,  
                lora_dropout=0.1
            )
        else:
            config = LoraConfig(
                target_modules=["out_proj",  "ffn.1", "fnn.3"],
                r=64,  
                lora_alpha=128,  
                lora_dropout=0.1
            )
        model = get_peft_model(model, config)
        checkpoint = torch.load(args.lora_ckpt, map_location='cpu')
        load_status = model.load_state_dict(checkpoint, strict=False)
        assert len(load_status.unexpected_keys) == 0, f"Unexpected keys in LoRA: {load_status.unexpected_keys}"
        print(f"Loaded LoRA from {args.lora_ckpt}")

    model = model.cuda()
    model.eval()

    for a, b in tqdm(zip(processed_data, triplet_prompts), total=len(processed_data), desc='Sampling'):
        sampled_data = sample((a, b, args.out_dir, model))


def detect_knot(pdb_path):
    open_chain = homfly(pdb_path, tries=200)
    knot_prob = 0
    knot_type = ''
    for k, v in open_chain.items():
        if v > knot_prob:
            knot_prob = v
            knot_type = k
    
    return knot_prob, knot_type


def detect_knot_in_csv(csv_path):
    """
    Detect knots in the generated designs.
    """
    df = pd.read_csv(csv_path)
    if 'knot_prob' in df.columns and 'knot_type' in df.columns:
        return
    dir_ = os.path.dirname(csv_path)
    pdb_paths = [os.path.join(dir_, f"{id_}.pdb") for id_ in df['id']]
    res = [detect_knot(pdb_path) for pdb_path in pdb_paths]
    knot_probs = [r[0] for r in res]
    knot_types = [r[1] for r in res]
    df['knot_prob'] = knot_probs
    df['knot_type'] = knot_types
    df.to_csv(csv_path, index=False)



def add_site_rmsd_in_csv(csv_path):
    """
    Add site RMSD to the generated designs.
    """
    df = pd.read_csv(csv_path)
    template_id = df['id'][0].split('_')[0]
    template_processed_file = os.path.join(args.processed_dir, f"{template_id}.npy")
    template = np.load(template_processed_file, allow_pickle=True).item()
    template = ESMProtein(sequence=template['sequence'], coordinates=torch.from_numpy(template['atom37_positions']))
    template_chain = template.to_protein_chain()

    try:
        motif_idx = df['motif_idx'][0].split(',')
        motif_idx = [int(i) for i in motif_idx]
    except:
        motif_idx = []
    if len(motif_idx) <= 1 or len(motif_idx) > len(template_chain.sequence) - 4:
        df['site_rmsd'] = np.nan
    else:
        for i, row in df.iterrows():
            design_path = os.path.join(os.path.dirname(csv_path), f"{row['id']}.pdb")
            design = ESMProtein.from_pdb(design_path)
            chain = design.to_protein_chain()
            assert len(chain.sequence) == len(template_chain.sequence), f"Length of design {row['id']} does not match template {template_id}"
            site_rmsd = chain[motif_idx].rmsd(template_chain[motif_idx])
            df.at[i, 'site_rmsd'] = site_rmsd
    df.to_csv(csv_path, index=False)


def add_site_rmsd(args):
        csv_files = os.listdir(args.out_dir)
        csv_paths = [os.path.join(args.out_dir, f) for f in csv_files if f.endswith('.csv')]
        csv_paths = sorted(csv_paths)
        print(f"Adding site rmsd info")
        with Pool(40) as p:
            list(tqdm(p.map(add_site_rmsd_in_csv, csv_paths), desc='Adding site rmsd info', total=len(csv_paths)))


def is_knot(row:pd.Series) -> bool:
    """
    Check if the design is a knot.
    """
    return row['knot_prob'] >= 0.7 and row['knot_type'] != '0_1'


def is_good_pair(row1:pd.Series, row2:pd.Series) -> bool:
    """
    Check if the pair is a good pair.
    """
    if is_knot(row1) and row1['ptm'] >= 0.8:
        if (row1['ptm'] - row2['ptm'] > 0.2) or not is_knot(row2):
            return {'rejected': row2, 'chosen': row1}

    if is_knot(row2) and row2['ptm'] >= 0.8:
        if (row2['ptm'] - row1['ptm'] > 0.2) or not is_knot(row1):
            return {'rejected': row1, 'chosen': row2}

    return None


def generate_preference_pairs(args):
    """
    Generate preference pairs for the generated designs.
    """
    result_files = os.listdir(args.out_dir)
    result_files = [f for f in result_files if f.endswith('.csv')]
    result_paths = [os.path.join(args.out_dir, f) for f in result_files]
    preference_pairs = []
    
    for path in tqdm(result_paths, desc='Generating preference pairs'):
        df = pd.read_csv(path)
        if df['motif_idx'].values[0] == '' or 'knot_prob' not in df.columns:
            continue
        # assert df['motif_idx'].nunique() == 1, f"Motif index should be the same for all designs in {path}"
        for i in range(len(df)):
            for j in range(i+1, len(df)):
                pair = is_good_pair(df.iloc[i], df.iloc[j])
                if pair is not None:
                    preference_pairs.append({
                        'motif_idx': df['motif_idx'][0],
                        'chosen': pair['chosen']['id'],
                        'rejected': pair['rejected']['id'],
                    })
    
    preference_pairs = pd.DataFrame(preference_pairs)
    
    print(f'Number of preference pairs: {len(preference_pairs)}')
    print(f'Number of prompts: {preference_pairs["motif_idx"].nunique()}')

    preference_pairs.to_csv(os.path.join(os.path.dirname(args.out_dir), 'preference_pairs_r256_130000_0.8.csv'), index=False)
    print(f'Saved preference pairs to {os.path.join(os.path.dirname(args.out_dir), "preference_pairs.csv")}')


def main(args):

    # gpu
    generate_design(args)

    # cpu
    # detect_knot_in_dir(args)
    # generate_preference_pairs(args)


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--lora_ckpt', type=str, help='Directory containing PDB files')
    parser.add_argument('--processed_dir', type=str, help='Directory containing processed PDB features')
    parser.add_argument('--out_dir', type=str, help='Directory to save preference data')
    parser.add_argument('--detect_knot', action='store_true', help='Detect knots in the generated designs')
    parser.add_argument('--range', type=int, nargs='+', default=[0, -1], help='Range of indices to sample from')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--pair', action='store_true')
    parser.add_argument('--add_site_rmsd', action='store_true', help='Add site RMSD to the generated designs')
    args = parser.parse_args()

    if args.add_site_rmsd:
        add_site_rmsd(args)
    elif args.detect_knot:
        csv_files = os.listdir(args.out_dir)
        csv_paths = [os.path.join(args.out_dir, f) for f in csv_files if f.endswith('.csv')]
        csv_paths = sorted(csv_paths)
        if args.range[1] > len(csv_paths) or args.range[1] == -1:
            args.range[1] = len(csv_paths)
        csv_paths = csv_paths[args.range[0]:args.range[1]]
        print(f"Detecting knots in from {args.range[0]} to {args.range[1]}")
        if args.reverse:
            csv_paths = csv_paths[::-1]
            print(f"Detecting knots in reverse order")
        for path in tqdm(csv_paths, desc='Detecting knots'):
            detect_knot_in_csv(path)
    elif args.pair:
        generate_preference_pairs(args)
    else:
        main(args)
