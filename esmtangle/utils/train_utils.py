from esm.utils.constants import esm3 as C
from esm.utils.sampling import _BatchedESMProteinTensor

import torch

import numpy as np
import math


mask_tokens = {
    "sequence": C.SEQUENCE_MASK_TOKEN,
    "structure": C.STRUCTURE_MASK_TOKEN,
    "coordinates": float('nan'),
}


def expand_triplet(triplet, L):
    """
    Expand triplet, determine whether to mask the left or right residue with 50% probability.
    Args:
        triplet: Close residue triplet, (3,)
        L: Length of the sequence + 2
    Returns:
        triplet: Expanded triplet, (3-9,)
    """
    assert triplet[2] < L-1, f' Triplet:{triplet} L:{L}'
    to_add = []
    for i in range(3):
        if torch.rand(1) < 0.5:
            to_add.append(triplet[i]-1)
        if torch.rand(1) < 0.5:
            to_add.append(triplet[i]+1)
    if len(to_add) > 0:
        triplet = torch.cat([triplet, torch.tensor(to_add, dtype=torch.int32)])
    assert 3 <= len(triplet) and len(triplet) <= 9, f' Triplet:{triplet} L:{L}'
    return triplet.to(torch.int32)


def sample_mask_rate(B, track):
    """
    Sample mask rate for a track.
    Returns:
        mask_rate: [B,], mask rate for each sample
    """
    u = torch.rand(B)
    if track == 'sequence':
        # 80% of the time, sample from β(3, 9) distribution, 20% from uniform
        mask_rate = torch.zeros(B)
        for i in range(B):
            r = np.random.beta(3, 9) if np.random.rand() < 0.8 else np.random.uniform(0, 1)
            mask_rate[i] = r
    elif track == 'structure':
        mask_rate = torch.cos(math.pi/2*u)
    elif track == 'coordinates':
        mask_rate = 1 - torch.pow(u, 3)
    else:
        raise ValueError(f"Unknown track: {track}")

    return mask_rate


def get_coordinate_mask(valid_pos: torch.Tensor, triplets: torch.Tensor, span_prob: float, triplet_prob: float, schecule:str = 'coordinates'):
    """
    Generate coordinate mask based on valid positions, 50% span masking, 50% random masking.
    Args:
        valid_pos (B, L): Boolean tensor indicating valid (non-padding) positions.
        triplets (B, 3): Close residue triplets.
        span_prob: Probability of span masking. Defaults to 0.5.
        triplet_prob: Probability of triplet masking. Defaults to 0.05.
    Returns:
        torch.Tensor: Boolean mask indicating which valid positions are masked.
    """
    mask = torch.zeros_like(valid_pos, dtype=torch.bool, device=valid_pos.device)

    B, L = valid_pos.size()
    mask_rates = sample_mask_rate(B, schecule)
    for i in range(B):

        if torch.rand(1) < triplet_prob:
            # Triplet masking
            if triplets[i].sum() > 0:   # Check if there are any triplets
                mask[i, expand_triplet(triplets[i], L)] = True
                mask[i] = ~ mask[i]  # Invert the mask
                continue 

        if torch.rand(1) < span_prob:
            # Span masking
            valid_indices = torch.where(valid_pos[i])[0]
            num_valid = len(valid_indices)

            num_masks = int(mask_rates[i] * num_valid)
            if num_masks <= 0:
                continue

            start_idx = torch.randint(0, len(valid_indices) - num_masks + 1, (1,)).item()
            span_indices = valid_indices[start_idx:start_idx + num_masks]

            mask[i, span_indices] = True
        else:
            # Random masking
            mask[i] = torch.rand(*(mask[i].size())) < mask_rates[i]  # [L,]

    return mask & valid_pos


def random_mask_and_drop(batch_tensor: _BatchedESMProteinTensor, triplets: torch.Tensor, dtype=None):
    """
    Randomly mask tokens or drop a track.
    Args:
        triplets: Close residue triplets, (N, 3)
    Returns:
        batch_tensor: Masked tensor
        mask: Dict of masks
    """
    B, L = batch_tensor.sequence.shape
    device = batch_tensor.sequence.device
    on_tracks = ['sequence', 'structure', 'coordinates']
    # [B, L], True if valid position
    valid_pos = (batch_tensor.sequence != C.SEQUENCE_PAD_TOKEN) &   \
                (batch_tensor.sequence != C.SEQUENCE_BOS_TOKEN ) &  \
                (batch_tensor.sequence != C.SEQUENCE_EOS_TOKEN )
    
    # [B, L], True if masked
    mask = {}
    for track in on_tracks:
        if getattr(batch_tensor, track) is None:
            raise ValueError(f"Track {track} is None")
        if track == 'coordinates':
            mask[track] = get_coordinate_mask(valid_pos, triplets, span_prob=0.5, triplet_prob=0.05)
        else:
            mask[track] = torch.rand(B, L, device=device) \
                < sample_mask_rate(B, track)[..., None].to(device)
            mask[track] &= valid_pos

    # Apply mask
    for track in on_tracks:
        new_tensor = getattr(batch_tensor, track)
        new_tensor[mask[track]] = mask_tokens[track]
        setattr(batch_tensor, track, new_tensor)

    # Randomly drop a whole track
    drop_structure = torch.rand(B, device=device) < 0.25
    batch_tensor.structure[drop_structure] = mask_tokens['structure']
    mask['structure'] = mask['structure'] | drop_structure[..., None] & valid_pos
    drop_coord = torch.rand(B, device=device) < 0.5
    batch_tensor.coordinates[drop_coord] = mask_tokens['coordinates']
    mask['coordinates'] = mask['coordinates'] | drop_coord[..., None] & valid_pos
    drop_seq = torch.rand(B, device=device) <= 1.0
    batch_tensor.sequence[drop_seq] = mask_tokens['sequence']
    mask['sequence'] = mask['sequence'] | drop_seq[..., None] & valid_pos
    if np.random.rand() < 0.5:
        batch_tensor.secondary_structure = torch.zeros_like(batch_tensor.secondary_structure, device=device) + C.SS8_PAD_TOKEN

    if dtype is not None:
        # batch_tensor.sequence = batch_tensor.sequence.to(dtype)
        # batch_tensor.structure = batch_tensor.structure.to(dtype)
        batch_tensor.coordinates = batch_tensor.coordinates.to(dtype)

    return batch_tensor, mask



def random_mask_and_drop_new(batch_tensor: _BatchedESMProteinTensor, triplets: torch.Tensor, dtype=None):
    """
    Randomly mask tokens or drop a track.
    Args:
        triplets: Close residue triplets, (N, 3)
    Returns:
        batch_tensor: Masked tensor
        mask: Dict of masks
    """
    B, L = batch_tensor.sequence.shape
    device = batch_tensor.sequence.device
    on_tracks = ['sequence', 'structure', 'coordinates']
    # [B, L], True if valid position
    valid_pos = (batch_tensor.sequence != C.SEQUENCE_PAD_TOKEN) &   \
                (batch_tensor.sequence != C.SEQUENCE_BOS_TOKEN ) &  \
                (batch_tensor.sequence != C.SEQUENCE_EOS_TOKEN )
    
    # [B, L], True if masked
    mask = {}
    mask['coordinates'] = get_coordinate_mask(valid_pos, triplets, span_prob=0.5, triplet_prob=0.1, schecule='coordinates')
    mask['sequence'] = mask['coordinates'].clone()
    mask['structure'] = mask['coordinates'].clone()

    # Apply mask
    for track in on_tracks:
        new_tensor = getattr(batch_tensor, track)
        new_tensor[mask[track]] = mask_tokens[track]
        setattr(batch_tensor, track, new_tensor)

    if dtype is not None:
        # batch_tensor.sequence = batch_tensor.sequence.to(dtype)
        # batch_tensor.structure = batch_tensor.structure.to(dtype)
        batch_tensor.coordinates = batch_tensor.coordinates.to(dtype)

    return batch_tensor, mask



if __name__ == "__main__":
    num_samples = 100000
    mask_rates = sample_mask_rate(num_samples, 'structure')
    plt.hist(mask_rates, bins=1000, density=True)
    plt.xlabel("Mask Rate")
    plt.ylabel("Density")
    plt.savefig("mask_rate_sequence.png")
    plt.close()