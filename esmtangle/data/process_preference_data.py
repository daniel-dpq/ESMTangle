import pandas as pd
import os
from multiprocessing import Pool
from esm.utils.structure.protein_chain import ProteinChain
import pickle as pkl
from esm.sdk.api import ESMProtein, ESMProteinTensor
import torch
import numpy as np
from tqdm import tqdm
torch.multiprocessing.set_sharing_strategy('file_system')

def pocess_row(motif_idx, chosen, rejected):
    try:
        # read a row from the preference data
        motif_idx = motif_idx.split(',')
        motif_idx = [int(idx) for idx in motif_idx]

        processed_file = os.path.join('/project/hggroup/puqing/esmknot/data/preference_data/preference_data_processed', chosen + '_' + rejected + '.pkl')
        if os.path.exists(processed_file) and os.stat(processed_file).st_size != 0:
            with open(processed_file, 'rb') as handle:
                data = pkl.load(handle)
                assert 'prompt' in data and 'rejected' in data and 'chosen' in data, "Processed data missing keys"
                assert isinstance(data['prompt'], ESMProtein) and isinstance(data['rejected'], ESMProtein) and isinstance(data['chosen'], ESMProtein), "Processed data has incorrect types"
                if len(data['rejected'].sequence) > 512:
                    os.remove(processed_file)  # Remove the file if the sequence is too long
                return
        print(f"Processing motif_idx: {motif_idx}, chosen: {chosen}, rejected: {rejected}")

        rejected_path = os.path.join('/project/hggroup/puqing/esmknot/data/preference_data/preference_data_pdb', f"{rejected}.pdb")
        chosen_path = os.path.join('/project/hggroup/puqing/esmknot/data/preference_data/preference_data_pdb', f"{chosen}.pdb")
        if not os.path.exists(rejected_path) or not os.path.exists(chosen_path):
            return

        rejected = ProteinChain.from_pdb(rejected_path, chain_id='A', is_predicted=False)
        if len(rejected.sequence) > 512:
            return
        chosen = ProteinChain.from_pdb(chosen_path, chain_id='A', is_predicted=False)
        assert len(rejected.sequence) == len(chosen.sequence), f"Rejected and chosen sequence length mismatch: {len(rejected.sequence)} vs {len(chosen.sequence)}"

        rejected = ESMProtein(
            sequence=rejected.sequence,
            coordinates=torch.from_numpy(rejected.atom37_positions),
        )
        chosen = ESMProtein(
            sequence=chosen.sequence,
            coordinates=torch.from_numpy(chosen.atom37_positions),
        )

        length = len(rejected.sequence)
        sequence_prompt = ["_"] * length
        structure_prompt = torch.full((length, 37, 3), np.nan)
        for idx in motif_idx:
            assert rejected.sequence[idx] == chosen.sequence[idx], f"Rejected and chosen sequence mismatch: {rejected.sequence[idx]} vs {chosen.sequence[idx]}"
            sequence_prompt[idx] = rejected.sequence[idx]
        structure_prompt[motif_idx] = rejected.coordinates[motif_idx]
        sequence_prompt = "".join(sequence_prompt)
        prompt = ESMProtein(sequence=sequence_prompt, coordinates=structure_prompt)

        data = {
            'prompt': prompt,
            'rejected': rejected,
            'chosen': chosen,
        }
        with open(processed_file, 'wb') as handle:
            pkl.dump(data, handle)

    except:
        print(f"Error processing motif_idx: {motif_idx}, chosen: {chosen}, rejected: {rejected}")


df = pd.read_csv('/project/hggroup/puqing/esmknot/data/preference_data/preference_data_merged.csv')
df = df[df['motif_idx'].apply(lambda x: isinstance(x, str))].reset_index(drop=True)

df = df.sample(frac=1).reset_index(drop=True)
motif_idx_list = df['motif_idx'].tolist()
chosen_list = df['chosen'].tolist()
rejected_list = df['rejected'].tolist()
# binary_names = [chosen + '_' + rejected + '.pkl' for chosen, rejected in zip(chosen_list, rejected_list)]
# not_exist_idx = [i for i, name in enumerate(binary_names) if not os.path.exists(os.path.join('/project/hggroup/puqing/esmknot/data/preference_data/preference_data_processed', name)) or os.stat(os.path.join('/project/hggroup/puqing/esmknot/data/preference_data/preference_data_processed', name)).st_size == 0]
# print(f"Number of rows to process: {len(motif_idx_list)}")
# print(f"Number of rows to process after filtering: {len(not_exist_idx)}")
# motif_idx_list = [motif_idx_list[i] for i in not_exist_idx]
# chosen_list = [chosen_list[i] for i in not_exist_idx]
# rejected_list = [rejected_list[i] for i in not_exist_idx]


# with Pool(28) as pool:
#     results = pool.starmap(pocess_row, zip(motif_idx_list, chosen_list, rejected_list))

for i in tqdm(range(len(motif_idx_list))):
    pocess_row(motif_idx_list[i], chosen_list[i], rejected_list[i])