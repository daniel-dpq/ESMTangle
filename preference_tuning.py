import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss as CELoss
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.optim import lr_scheduler

from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient
from esm.utils.sampling import _BatchedESMProteinTensor
from esm.utils.generation import _stack_protein_tensors
from esm.utils.constants import esm3 as C

from peft import LoraConfig, TaskType, get_peft_model

from multiprocessing import cpu_count
from typing import Callable, List
from time import time
import datetime
from copy import deepcopy
import numpy as np
import random
import logging
import math
import os

import hydra
from omegaconf import DictConfig
import wandb

from esmknot.data.data_loader import IRPODataset, ESMProteinBatchCollator, IRPOBatchCollator
import esmknot.utils.train_utils as tu

from huggingface_hub import login
hf_token = "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"  
# login(token=hf_token)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)

def exists(x):
    return x is not None

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def cycle(dl):
    while True:
        for data in dl:
            yield data


class ESM3IRPOTrainer(object):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()

        self.data_cfg = cfg.data
        self.exp_cfg = cfg.experiment
        self.model_cfg = cfg.model

        # setup DDP
        assert torch.cuda.is_available(), "Training currently requires at least one GPU."
        self._init_ddp()

        # seed all
        self.seed_all()

        # for logging results in a folder periodically
        if self.rank == 0:
            os.makedirs(self.exp_cfg.log_dir, exist_ok = True)
            self.logger = self._create_logger(self.exp_cfg.log_dir)
        else:
            self.logger = self._create_logger(None)

        # model 
        self.model_name = self.model_cfg.name
        self.model: ESM3InferenceClient = ESM3.from_pretrained(self.model_name)

        # load checkpoint
        if self.exp_cfg.start_ckpt:
            config = LoraConfig(
                target_modules=["out_proj",  "ffn.1", "fnn.3", 'structure_head.0', 'structure_head.3'],
                r=256,
                lora_alpha=512,  
                lora_dropout=0.1,
            )
            self.model = get_peft_model(self.model, config)
            self.load_lora(self.exp_cfg.start_ckpt)
            self.model = self.model.merge_and_unload()
            if hasattr(self.model, 'peft_config'):
                del self.model.peft_config
            if hasattr(self.model, 'modules_to_save'):
                del self.model.modules_to_save
            self.start_step = 0
            self.logger.info(f"Loaded and merged LoRA checkpoint from {self.exp_cfg.start_ckpt}")
        else:
            raise ValueError("No strating LoRA checkpoint provided.")

        # setup reference model
        self.ref_model = deepcopy(self.model)
        self.ref_model.eval()
        self.ref_model = self.ref_model.to(self.device)
        for name, param in self.ref_model.named_parameters():
            param.requires_grad = False

        # Setup LoRA
        if self.model_cfg.use_lora:
            config = LoraConfig(
                target_modules=["out_proj",  "ffn.1", "fnn.3"],
                r=self.model_cfg.lora.r,
                lora_alpha=self.model_cfg.lora.alpha,  
                lora_dropout=self.model_cfg.lora.dropout
            )
            self.model = get_peft_model(self.model, config)
            if self.exp_cfg.lora_ckpt:
                self.load_lora(self.exp_cfg.lora_ckpt)
            if self.rank == 0:
                self.model.print_trainable_parameters()
        else:
            for name, param in self.model.named_parameters():
                if 'output_heads' in name and 'structure_head' not in name:
                    param.requires_grad = False
            
        self.model = DDP(self.model.to(self.device), device_ids = [self.rank])
        self.logger.info(f"ESM3 Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # setup data
        self.load_data(is_train=True)

        # optimizer and loss
        self.optim = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.exp_cfg.lr, 
            betas=self.exp_cfg.adamW_betas,
            weight_decay=self.exp_cfg.adamW_weight_decay
        )
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optim, 
            T_max=self.exp_cfg.epochs * len(self.train_dataloader) // self.exp_cfg.accumulate_steps,
            eta_min=0.1*self.exp_cfg.lr, 
            last_epoch=-1
        )

        # mixed precision training
        self.mixed_precision = self.exp_cfg.mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # wandb
        if self.exp_cfg.use_wandb and self.rank == 0:
            self._init_wandb()
    
    def _init_ddp(self):
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=18000))
        self.world_size = dist.get_world_size()
        assert self.exp_cfg.global_batch_size % self.world_size == 0, f"Batch size must be divisible by world size."
        self.rank = dist.get_rank()
        self.device = self.rank % torch.cuda.device_count()
        self.seed = self.data_cfg.seed * self.world_size + self.rank
        torch.manual_seed(self.seed)
        torch.cuda.set_device(self.device)
        logging.info(f"Starting rank={self.rank}, seed={self.seed}, world_size={self.world_size}.")

    def _init_wandb(self):
        wandb.init(
            project=self.exp_cfg.project,
            name=self.exp_cfg.name,
            dir=self.exp_cfg.log_dir,
        )
        self.logger.info(f"Initialized wandb project: {self.exp_cfg.project}, run: {wandb.run.id}")
    
    def seed_all(self):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def requires_grad(self, model, flag=True):
        """
        Set requires_grad flag for all parameters in a model.
        """
        for p in model.parameters():
            p.requires_grad = flag
    
    def cleanup(self):
        """
        End DDP training.
        """
        dist.destroy_process_group()

    def _create_logger(self, logging_dir):
        """
        Create a logger that writes to a log file and stdout.
        """
        if self.rank == 0:  # real logger
            logging.basicConfig(
                level=logging.INFO,
                format='[\033[34m%(asctime)s\033[0m] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=[logging.StreamHandler()]
            )
            logger = logging.getLogger(__name__)
        else:  # dummy logger (does nothing)
            logging.info(f"Rank {self.rank} does not create a logger.")
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.CRITICAL)
            logger.addHandler(logging.NullHandler())
        return logger

    def load_data(self, is_train):
        dataset = IRPODataset(self.data_cfg, self.logger)
        val_size = int(len(dataset) * self.data_cfg.val_ratio)
        train_size = len(dataset) - val_size
        self.train_set, self.val_set = random_split(
            dataset, 
            [train_size, val_size], 
            generator=torch.Generator().manual_seed(self.data_cfg.seed)
        )
        self.logger.info(f"Train set size: {len(self.train_set)}, Val set size: {len(self.val_set)}")
        collator = IRPOBatchCollator(self.data_cfg, self.logger)
        torch.manual_seed(2)
        torch.cuda.manual_seed_all(2)
        self.sampler = DistributedSampler(
            self.train_set,
            num_replicas=self.world_size,
            rank=self.rank,
            drop_last=True,
            shuffle=is_train,
            seed=self.data_cfg.seed
        )
        self.train_dataloader = DataLoader(
            self.train_set,
            batch_size=int(self.exp_cfg.global_batch_size // self.world_size),
            shuffle=False,
            sampler=self.sampler,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
            collate_fn=collator
        )
        self.val_dataloader = DataLoader(
            self.val_set,
            batch_size=int(self.exp_cfg.global_batch_size // self.world_size),
            shuffle=False,
            sampler=None,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collator
        )
    
    def batch_encode(self, batch):
        """
            Batch:
            [{'prompt': ESMProtein, 'rejected': ESMProtein, 'chosen': ESMProtein}, ...]
            Return:
            {'prompt': _BatchedESMProteinTensor, 'rejected': _BatchedESMProteinTensor, 'chosen': _BatchedESMProteinTensor}
        """
        with torch.no_grad():
            keys = ['prompt', 'rejected', 'chosen']
            batched_dict = {}
            for k in keys:
                batch_tensor = [
                    self.ref_model.encode(sample[k])
                    for sample in batch
                ] 
                sequence_lengths = [len(p.sequence) for p in batch_tensor]
                batched_dict[k] = _stack_protein_tensors(
                    batch_tensor, sequence_lengths, 
                    self.model.module.tokenizers, 
                    self.model.module.device,
                )

        assert batched_dict['prompt'].sequence.shape == batched_dict['rejected'].sequence.shape, f"Prompt and rejected sequence shape mismatch: {batched_dict['prompt'].sequence.shape} vs {batched_dict['rejected'].sequence.shape}"
        assert batched_dict['prompt'].sequence.shape == batched_dict['chosen'].sequence.shape, f"Prompt and chosen sequence shape mismatch: {batched_dict['prompt'].sequence.shape} vs {batched_dict['chosen'].sequence.shape}"
        return batched_dict

    def save(self, milestone):
        checkpoint = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        checkpoint_path = os.path.join(self.exp_cfg.log_dir, f'model-{milestone}.pt')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def save_lora(self, milestone):
        checkpoint = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        lora_checkpoint = {
            k: v for k, v in checkpoint.items() if 'lora' in k
        }
        checkpoint_path = os.path.join(self.exp_cfg.log_dir, f'lora-{milestone}.pt')
        torch.save(lora_checkpoint, checkpoint_path)
        self.logger.info(f"Saved LoRA checkpoint to {checkpoint_path}")

    def load(self, checkpoint_path, strict=True):
        'load right after the model is initialized'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'], strict=strict)
        else:
            self.model.load_state_dict(checkpoint, strict=strict)
        self.start_step = int(checkpoint_path.split('-')[-1].split('.')[0])
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}, strict={strict}")

    def load_lora(self, checkpoint_path):
        'load right after the the lora is embedded'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        load_status = self.model.load_state_dict(checkpoint, strict=False)
        assert len(load_status.unexpected_keys) == 0, f"Unexpected keys in LoRA: {load_status.unexpected_keys}"
        self.start_step = int(checkpoint_path.split('-')[-1].split('.')[0])
        self.logger.info(f"Loaded LoRA from {checkpoint_path}")

    def _write_to_fasta(self, seqs: List[str], path: str):
        with open(path, 'w') as f:
            for i, seq in enumerate(seqs):
                f.write(f'>seq_{i}\n')
                f.write(f'{seq}\n')
    
    def sample_y_mask(self, valid_pos: torch.Tensor, motif_pos: torch.Tensor ,span_prob: float, schecule:str = 'coordinates'):
        """
        Generate coordinate mask based on valid positions, 50% span masking, 50% random masking.
        Args:
            valid_pos (B, L): Boolean tensor indicating valid (non-padding/bos/eos) positions.
            span_prob: Probability of span masking. Defaults to 0.5.
        Returns:
            torch.Tensor: Boolean mask indicating which valid positions are masked.
        """
        mask = torch.zeros_like(valid_pos, dtype=torch.bool, device=valid_pos.device)

        B, L = valid_pos.size()
        mask_rates = tu.sample_mask_rate(B, schecule)
        mask_rates = torch.clamp(mask_rates, min=0.5, max=1)
        for i in range(B):
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
            
            if mask[i][~motif_pos[i]].sum() == 0:
                # Ensure at least one position is masked
                mask[i] = True

        return mask & valid_pos

    def per_token_logps_from_logits(self, logits, labels):
        # [B, L+2, NUM_CLASS], [B, L+2] -> [B, L+2]
        pre_token_logps = torch.gather(logits.log_softmax(-1), dim=-1, index=labels.clone().unsqueeze(-1)).squeeze(-1)
        return pre_token_logps
    
    def all_logps_from_logits(self, logits, labels, loss_mask):
        labels[loss_mask] = 0.
        per_token_logps = self.per_token_logps_from_logits(logits, labels)
        per_token_logps[loss_mask] = 0.
        # [B, L+2] -> [B, ]
        all_logps = per_token_logps.sum(-1)
        valid_num = (~loss_mask).float().sum(-1)
        mean_logps = all_logps / valid_num
        return all_logps, mean_logps

    def position_kl_from_logits(self, logits, ref_logits, loss_mask):
        # [B, L+2, NUM_CLASS]
        vocab_logps = logits.log_softmax(-1)
        ref_vocab_ps = ref_logits.softmax(-1)
        ref_vocab_logps = ref_logits.log_softmax(-1)

        # [B, L+2, NUM_CLASS] -> [B, L+2]
        per_position_kl = (ref_vocab_ps * (ref_vocab_logps - vocab_logps)).sum(-1)
        loss_mask = (~loss_mask).float()

        # [B, L+2] -> [B, ]
        all_position_kl = (per_position_kl * loss_mask).sum(-1)
        return all_position_kl

    def compute_loss(self, batched_dict: dict[str, _BatchedESMProteinTensor]):

        alpha = self.exp_cfg.loss_param.alpha
        beta = self.exp_cfg.loss_param.beta
        
        # [B, L+2], True for unmasked position, e.g. <bos>, <eos>, <pad>, motif
        non_mask_pos = (batched_dict['prompt'].sequence != C.SEQUENCE_MASK_TOKEN)
        motif_pos = (batched_dict['prompt'].sequence != C.SEQUENCE_BOS_TOKEN) &   \
                    (batched_dict['prompt'].sequence != C.SEQUENCE_PAD_TOKEN ) &  \
                    (batched_dict['prompt'].sequence != C.SEQUENCE_EOS_TOKEN ) & \
                    (batched_dict['prompt'].sequence != C.SEQUENCE_MASK_TOKEN)
        mask_pos = batched_dict['prompt'].sequence == C.SEQUENCE_MASK_TOKEN
        termin_pos = (batched_dict['prompt'].sequence == C.SEQUENCE_EOS_TOKEN) | \
                     (batched_dict['prompt'].sequence == C.SEQUENCE_BOS_TOKEN) | \
                     (batched_dict['prompt'].sequence == C.SEQUENCE_PAD_TOKEN)

        # include partial y into the prompt
        y_mask = self.sample_y_mask(motif_pos | mask_pos, motif_pos, span_prob=0.5, schecule='coordinates') # True for masked position
        loss_mask = (~y_mask) | non_mask_pos

        # sequence_input
        input_sequence = torch.full_like(batched_dict['prompt'].sequence, C.SEQUENCE_MASK_TOKEN, dtype=batched_dict['prompt'].sequence.dtype, device=self.device)
        input_sequence[non_mask_pos] = batched_dict['prompt'].sequence[non_mask_pos]

        # chosen_input
        chosen_input_structure = torch.where(loss_mask, batched_dict['chosen'].structure, C.STRUCTURE_MASK_TOKEN)
        chosen_input_coordinates = torch.where(loss_mask.unsqueeze(-1).unsqueeze(-1), batched_dict['chosen'].coordinates, batched_dict['prompt'].coordinates)

        # reject input
        rejected_input_structure = torch.where(loss_mask, batched_dict['rejected'].structure, C.STRUCTURE_MASK_TOKEN)
        rejected_input_coordinates = torch.where(loss_mask.unsqueeze(-1).unsqueeze(-1), batched_dict['rejected'].coordinates, batched_dict['rejected'].coordinates)

        chosen_out = self.model(
            sequence_tokens=input_sequence,
            structure_tokens=chosen_input_structure,
            structure_coords=chosen_input_coordinates,
        )

        rejected_out = self.model(
            sequence_tokens=input_sequence,
            structure_tokens=rejected_input_structure,
            structure_coords=rejected_input_coordinates,
        )

        chosen_ref_out = self.ref_model(
            sequence_tokens=input_sequence,
            structure_tokens=chosen_input_structure,
            structure_coords=chosen_input_coordinates,
        )

        rejected_ref_out = self.ref_model(
            sequence_tokens=input_sequence,
            structure_tokens=rejected_input_structure,
            structure_coords=rejected_input_coordinates,
        )

        # loss_mask = non_mask_pos
        # out = self.model(
        #     sequence_tokens=batched_dict['prompt'].sequence,
        #     structure_tokens=batched_dict['prompt'].structure,
        #     structure_coords=batched_dict['prompt'].coordinates,
        # )
        # ref_out = self.ref_model(
        #     sequence_tokens=batched_dict['prompt'].sequence,
        #     structure_tokens=batched_dict['prompt'].structure,
        #     structure_coords=batched_dict['prompt'].coordinates,
        # )
        # chosen_out = out
        # rejected_out = out
        # chosen_ref_out = ref_out
        # rejected_ref_out = ref_out

        # [B, L+2, NUM_CLASS]
        chosen_logits = chosen_out.structure_logits
        rejected_logits = rejected_out.structure_logits
        chosen_ref_logits = chosen_ref_out.structure_logits
        rejected_ref_logits = rejected_ref_out.structure_logits

        # [B, L+2]
        rejected_labels = batched_dict['rejected'].structure
        chosen_labels = batched_dict['chosen'].structure
        
        # [B, ]
        policy_rejected_logps, mean_p_rejected_logps = self.all_logps_from_logits(rejected_logits, rejected_labels, loss_mask)
        policy_chosen_logps, mean_p_chosen_logps = self.all_logps_from_logits(chosen_logits, chosen_labels, loss_mask)
        ref_rejected_logps, _ = self.all_logps_from_logits(rejected_ref_logits, rejected_labels, loss_mask)
        ref_chosen_logps, _ = self.all_logps_from_logits(chosen_ref_logits, chosen_labels, loss_mask)

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        dpo_loss = - F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
        
        if self.exp_cfg.loss == 'dpo':
            return dpo_loss, mean_p_chosen_logps.mean(), mean_p_rejected_logps.mean()
        elif self.exp_cfg.loss == 'irpo':
            nll = F.cross_entropy(chosen_logits[~loss_mask], chosen_labels[~loss_mask])
            irpo_loss = nll + alpha * dpo_loss
            return irpo_loss, mean_p_chosen_logps.mean(), mean_p_rejected_logps.mean()
        elif self.exp_cfg.loss == 'tdpo':
            chosen_position_kl = self.position_kl_from_logits(chosen_logits, chosen_ref_logits, loss_mask)
            rejected_position_kl = self.position_kl_from_logits(rejected_logits, rejected_ref_logits, loss_mask)
            tdpo_loss = - F.logsigmoid(beta * (pi_logratios - ref_logratios - alpha * (rejected_position_kl - chosen_position_kl.detach()))).mean()
            return tdpo_loss, mean_p_chosen_logps.mean(), mean_p_rejected_logps.mean()

    def train(self):
        # Prepare models for training:
        self.model.train() 

        # Variables for monitoring/logging purposes:
        train_steps = self.start_step
        log_steps = 0
        running_loss, accumulated_chosen_logp, accumulated_rejected_logp = 0, 0, 0
        start_time = time()

        self.logger.info(f"{self.exp_cfg.loss} Training for {self.exp_cfg.epochs} epochs...")
        self.logger.info(f'effective batch size: {self.exp_cfg.global_batch_size * self.exp_cfg.accumulate_steps}')
        for epoch in range(self.exp_cfg.epochs):
            self.sampler.set_epoch(epoch)

            for step, batch in enumerate(self.train_dataloader): 
                batched_dict = self.batch_encode(batch)
                if step % self.exp_cfg.accumulate_steps != 0:
                    with self.model.no_sync():
                        with torch.cuda.amp.autocast(enabled=self.mixed_precision, dtype=torch.bfloat16):
                            loss, chosen_logp, rejected_logp = self.compute_loss(batched_dict)
                        loss = loss / self.exp_cfg.accumulate_steps
                        self.scaler.scale(loss).backward()

                else:
                    with torch.cuda.amp.autocast(enabled=self.mixed_precision, dtype=torch.bfloat16):
                        loss, chosen_logp, rejected_logp = self.compute_loss(batched_dict)
                    loss = loss / self.exp_cfg.accumulate_steps
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.exp_cfg.max_grad_norm)
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    self.optim.zero_grad()
                    self.scheduler.step()

                torch.cuda.empty_cache()
                train_steps += 1
                log_steps += 1
                running_loss += loss.item()*self.exp_cfg.accumulate_steps
                accumulated_chosen_logp += chosen_logp.item()
                accumulated_rejected_logp += rejected_logp.item()
                if train_steps % self.exp_cfg.log_interval == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=self.device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    avg_chosen_logp = accumulated_chosen_logp / log_steps
                    avg_rejected_logp = accumulated_rejected_logp / log_steps
                    self.logger.info(f"[epoch={(epoch+1):03d}, step={train_steps:07d}] Train Loss: {avg_loss:.4f}, Chosen LogP: {avg_chosen_logp:.4f}, Rejected LogP: {avg_rejected_logp:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    # Log to wandb:
                    if self.exp_cfg.use_wandb and self.rank == 0:
                        wandb.log({
                            'train_loss': avg_loss,
                            'train_steps_per_sec': steps_per_sec,
                            'Chosen LogP': avg_chosen_logp,
                            'Rejected LogP': avg_rejected_logp,
                        }, step=train_steps)
                    # Reset monitoring variables:
                    running_loss, accumulated_chosen_logp, accumulated_rejected_logp = 0, 0, 0
                    log_steps = 0
                    start_time = time()

                # Save checkpoint:
                if train_steps % self.exp_cfg.save_and_eval_interval == 0:
                    if self.rank == 0:
                        self.eval(train_steps)
                        self.save_lora(train_steps)
    
    def eval(self, train_step):
        self.model.eval()
        accumulated_loss, accumulated_chosen_logp, accumulated_rejected_logp = 0, 0, 0
        start_time = time()
        
        for step, batch in enumerate(self.val_dataloader):
            batched_dict = self.batch_encode(batch)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.mixed_precision, dtype=torch.bfloat16):
                    loss, chosen_logp, rejected_logp = self.compute_loss(batched_dict)
                accumulated_loss += loss.item()
                accumulated_chosen_logp += chosen_logp.item()
                accumulated_rejected_logp += rejected_logp.item()

        avg_loss = accumulated_loss / (step+1)
        avg_chosen_logp = accumulated_chosen_logp / (step+1)
        avg_rejected_logp = accumulated_rejected_logp / (step+1)
        self.logger.info(f"[step={train_step:07d}] Val Loss: {avg_loss:.4f}, Val Chosen LogP: {avg_chosen_logp:.4f}, Val Rejected LogP: {avg_rejected_logp:.4f}, time: {time()-start_time:.2f}s")
        if self.exp_cfg.use_wandb:
            wandb.log({
                'val_loss': avg_loss,
                'val_chosen_logp': avg_chosen_logp,
                'val_rejected_logp': avg_rejected_logp,
            }, step=train_step)
        
        self.model.train()

    def cleanup(self):
        """
        End DDP training.
        """
        dist.destroy_process_group()


@hydra.main(config_path='esmknot/config', config_name='preference_tuning', version_base=None)
def main(cfg):
    trainer = ESM3IRPOTrainer(cfg)
    trainer.train()
    trainer.cleanup()


if __name__ == "__main__":
    main()
