from esm.utils.constants import esm3 as C
from esm.utils.generation import _stack_protein_tensors
from esm.sdk.api import ESMProtein, ESMProteinTensor
from esm.utils.sampling import _BatchedESMProteinTensor
from esm.utils.structure.protein_chain import ProteinChain

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset

import pandas as pd
import numpy as np
import pickle as pkl
import os
from typing import List, Dict
from tqdm import tqdm


class ESMProteinDataset(Dataset):
    def __init__(self, cfg, logger, is_train, dtype=torch.float32):
        
        self.cfg = cfg
        self.logger = logger
        self.is_train = is_train
        self.dtype = dtype

        df = pd.read_csv(self.cfg.data_path)
        if self.cfg.min_chain_plddt is not None:
            df = df[df['pLDDT_chain'] >= self.cfg.min_chain_plddt]
        if self.cfg.max_chain_length is not None:
            df = df[df['Chain_length'] <= self.cfg.max_chain_length]
        if self.cfg.max_loop_percent is not None:
            df = df[df['coil_percent'] <= self.cfg.max_loop_percent]
        if self.cfg.min_knot_prob is not None:
            df = df[df['Knot_probability'] >= self.cfg.min_knot_prob]
        df = df[df['Uniprot'].apply(lambda x: os.path.exists(os.path.join(self.cfg.feat_dir, f"{x}.npy")))]
        self.df = df

        self.grouped_ids = self._read_clusters()
        self.knotcore_range = self._read_knot_core_range()

        logger.info(f"Read Number of sequences: {len(self.df)}, Number of clusters: {len(self.grouped_ids)}")

    def __len__(self):
        return len(self.grouped_ids)

    def _read_clusters(self):
        if self.is_train:
            cluster = pd.read_csv(self.cfg.cluster_path, sep='\t', header=None, names=['rep_id', 'seq_id'])
            cluster = cluster[cluster['seq_id'].isin(self.df['Uniprot'])]
            grouped_ids = []
            for rep_id in cluster['rep_id'].unique():
                seq_ids = cluster[cluster['rep_id'] == rep_id]['seq_id'].values.tolist()
                if len(seq_ids) >= self.cfg.min_cluster_size:
                    grouped_ids.append(seq_ids)
        else:   # each sequence is a cluster
            grouped_ids = [[seq_id] for seq_id in self.df['Uniprot'].values]
        return grouped_ids
    
    def _read_knot_core_range(self):
        knotcore_range = {}
        for _, row in self.df.iterrows():
            uniprot_id = row['Uniprot']
            range_ = np.array(list(map(int, row['Knotcore_range'].split('-')))) - 1
            knotcore_range[uniprot_id] = range_
        return knotcore_range
    
    def __getitem__(self, idx):

        cluster = self.grouped_ids[idx]
        id_ = np.random.choice(cluster)
        feat_path = os.path.join(self.cfg.feat_dir, f"{id_}.npy")
        feat_dict = np.load(feat_path, allow_pickle=True).item()
        knotcore_range = self.knotcore_range[id_]
        L = len(feat_dict['sequence'])
        if knotcore_range[1] > L - 1 or knotcore_range[0] < 0:
            knotcore_range = None
        else:
            # H denotes the knotcore region, G denotes the rest
            knotcore_range = 'G' * knotcore_range[0] + 'H' * (knotcore_range[1] - knotcore_range[0] + 1) + 'G' * (L - knotcore_range[1] - 1)
        feat_dict['knotcore_range'] = knotcore_range

        return feat_dict


class ESMProteinBatchCollator(object):
    def __init__(self, cfg, logger, dtype=torch.float32, device='cuda'):
        self.cfg = cfg
        self.logger = logger
        self.dtype = dtype
        self.device = device
    
    def _extract_dict_list(self, dict_list):
        keys = dict_list[0].keys()
        dict_ = {}
        for key in keys:
            dict_[key] = [d[key] for d in dict_list]
        return dict_

    def _pad_dict(self, dict_):
        for key in dict_:
            if key == 'sequence':
                dict_[key] = self._pad_list(dict_[key], C.SEQUENCE_PAD_TOKEN)
            elif key == 'atom37_positions':
                dict_[key] = self._pad_list(dict_[key], C.STRUCTURE_PAD_TOKEN)
    
    def _pad_list(self, list_, pad_value):
        # given a list of numpy arrays, pad them to max_len
        max_len = max(len(l) for l in list_)
        tensor = torch.full((len(list_), max_len), pad_value)
        for i, l in enumerate(list_):
            tensor[i, :len(l)] = torch.from_numpy(l)
        return tensor
    
    def __call__(self, batch):
        return batch


def encode_protein(model, batch: List[Dict]):
    """
        Return:
            Sequence tensor: (B, L)
            Structure tensor: (B, L)
            Coordinate tensor: (B, L, 37, 3)
            plddt tensor: (B, L)
            triplet tensor: (B, 3)
    """
    def _pad_list(list_, pad_value):
        # given a list of numpy arrays, pad them to max_len
        max_len = max(len(l) for l in list_)
        tensor = torch.full((len(list_), max_len), pad_value)
        for i, l in enumerate(list_):
            tensor[i, :len(l)] = torch.from_numpy(l)
        return tensor

    model.eval()
    
    batch_protein = [
        ESMProtein(
            sequence=feat_dict['sequence'],
            coordinates=torch.from_numpy(feat_dict['atom37_positions']),
            secondary_structure=feat_dict['knotcore_range'],
        )
        for feat_dict in batch
    ]
    batch_tensor = [
        model.module.encode(protein)
        for protein in batch_protein
    ]

    sequence_lengths = [len(p.sequence) for p in batch_tensor]
    batch_tensor = _stack_protein_tensors(batch_tensor, sequence_lengths, model.module.tokenizers, model.module.device)

    plddts = [
        np.concatenate([[0.], feat_dict['confidence'], [0.]]) 
        for feat_dict in batch
    ]
    plddts = _pad_list(plddts, 0.) / 100.

    triplets = [
        feat_dict['triplets'][np.random.randint(0, len(feat_dict['triplets']))] if len(feat_dict['triplets']) > 0 else np.zeros(3)
        for feat_dict in batch
    ]
    triplets = torch.from_numpy(np.array(triplets)).to(torch.int32)

    return batch_tensor, plddts, triplets



class IRPODataset(Dataset):
    def __init__(self, cfg, logger, dtype=torch.float32):
        
        self.cfg = cfg
        self.logger = logger
        self.dtype = dtype

        self.df = pd.read_csv(self.cfg.preference_csv_path)
        self.df = self.df[self.df['motif_idx'].apply(lambda x: isinstance(x, str))].reset_index(drop=True)

        if dist.get_rank() == 0:
            loop = tqdm(self.df.iterrows(), total=self.df.shape[0])
        else:
            loop = self.df.iterrows()
        
        self.preference_data_paths = {
            self.process_row(row): row for _, row in loop
        }
        valid_idx = [idx for idx, path in enumerate(self.preference_data_paths) if path is not None]
        self.df = self.df.iloc[valid_idx].reset_index(drop=True)
        self.preference_data_paths = [
            path for path in self.preference_data_paths if path is not None
        ]
        assert len(self.df) == len(self.preference_data_paths), f"Dataframe length {len(self.df)} and preference data length {len(self.preference_data_paths)} mismatch"

        logger.info(f"Read Number of prefernece data: {len(self.preference_data_paths)}")

    def __len__(self):
        return len(self.preference_data_paths)

    def process_row(self, row):
        processed_file = os.path.join(self.cfg.preference_processed_dir, row['chosen'] + '_' + row['rejected'] + '.pkl')
        if os.path.exists(processed_file) and os.stat(processed_file).st_size != 0:
            return processed_file
        return None
    
    def __getitem__(self, idx):
        '''
        A dict with keys: prompt, rejected, chosen, all ESMProtein objects
        '''
        return self.preference_data_paths[idx]
    
    def split_eval(self, eval_fraction):
        """
        Split the dataset into train and eval sets
        """
        assert 0 < eval_fraction < 1, "eval_fraction should be between 0 and 1"
        prompts = self.df['motif_idx'].unique()
        prompt_num = len(prompts)
        num_eval = int(prompt_num * eval_fraction)
        eval_prompts = np.random.choice(prompts, size=num_eval, replace=False)
        eval_idx = self.df[self.df['motif_idx'].isin(eval_prompts)].index
        train_idx = np.setdiff1d(np.arange(len(self.df)), eval_idx)

        return [Subset(self, train_idx), Subset(self, eval_idx)]


class IRPOBatchCollator(object):
    def __init__(self, cfg, logger, dtype=torch.float32, device='cuda'):
        self.cfg = cfg
        self.logger = logger
        self.dtype = dtype
        self.device = device
    
    def _load_pkl(self, path):
        with open(path, 'rb') as f:
            return pkl.load(f)
    
    def __call__(self, batch):
        """
        batch: List[Dict]
        """
        # Load the preference data from the paths
        batch = [self._load_pkl(data_path) for data_path in batch]

        return batch