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

from peft import LoraConfig, TaskType, get_peft_model

from multiprocessing import cpu_count
from typing import Callable, List
from time import time
from copy import deepcopy
import numpy as np
import random
import logging
import math
import os

import hydra
from omegaconf import DictConfig
import wandb

from esmknot.data.data_loader import ESMProteinDataset, ESMProteinBatchCollator, encode_protein
import esmknot.utils.train_utils as tu

from huggingface_hub import login
hf_token = "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"  
login(token=hf_token)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


class ESM3Trainer(object):
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
 
        # LoRA
        if self.model_cfg.lora.enabled:
            config = LoraConfig(
                # task_type=TaskType.CAUSAL_LM,
                target_modules=["out_proj", "ffn.1", "fnn.3", 'structure_head.0', 'structure_head.3'],
                r=self.model_cfg.lora.r,  
                lora_alpha=self.model_cfg.lora.alpha,  
                lora_dropout=self.model_cfg.lora.dropout
            )
            self.model = get_peft_model(self.model, config)
            if self.rank == 0:
                self.model.print_trainable_parameters()
        else:
            for name, param in self.model.named_parameters():
                if 'head' in name and 'structure' not in name:
                    param.requires_grad = False
        
        # load checkpoint
        if self.exp_cfg.ckpt:
            self.load(self.exp_cfg.ckpt, strict=True)
        elif self.exp_cfg.lora_ckpt:
            self.load_lora(self.exp_cfg.lora_ckpt)
        else:
            self.start_step = 0
        self.model = DDP(self.model.to(self.device), device_ids = [self.rank])
        self.logger.info(f"ESM3 Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # optimizer and loss
        self.loss_fn = CELoss().to(self.device)
        self.optim = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.exp_cfg.lr, 
            betas=self.exp_cfg.adamW_betas,
            weight_decay=self.exp_cfg.adamW_weight_decay
        )
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=self.exp_cfg.epochs, eta_min=0, last_epoch=-1)

        # setup data
        self.load_data(is_train=True)

        # mixed precision training
        self.mixed_precision = self.exp_cfg.mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # wandb
        if self.exp_cfg.use_wandb and self.rank == 0:
            self._init_wandb()
    
    def _init_ddp(self):
        dist.init_process_group("nccl")
        assert self.exp_cfg.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
        self.rank = dist.get_rank()
        self.device = self.rank % torch.cuda.device_count()
        self.seed = self.data_cfg.seed * dist.get_world_size() + self.rank
        torch.manual_seed(self.seed)
        torch.cuda.set_device(self.device)
        logging.info(f"Starting rank={self.rank}, seed={self.seed}, world_size={dist.get_world_size()}.")

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
        dataset = ESMProteinDataset(self.data_cfg, self.logger, is_train=is_train)
        val_size = int(len(dataset) * self.data_cfg.val_ratio)
        train_size = len(dataset) - val_size
        self.train_set, self.val_set = random_split(
            dataset, 
            [train_size, val_size], 
            generator=torch.Generator().manual_seed(self.data_cfg.seed)
        )
        self.logger.info(f"Train set size: {len(self.train_set)}, Val set size: {len(self.val_set)}")
        collator = ESMProteinBatchCollator(self.data_cfg, self.logger)
        self.sampler = DistributedSampler(
            self.train_set,
            num_replicas=dist.get_world_size(),
            rank=self.rank,
            shuffle=is_train,
            seed=self.data_cfg.seed
        )
        self.train_dataloader = DataLoader(
            self.train_set,
            batch_size=int(self.exp_cfg.global_batch_size // dist.get_world_size()),
            shuffle=False,
            sampler=self.sampler,
            num_workers=dist.get_world_size(),
            pin_memory=True,
            drop_last=True,
            collate_fn=collator
        )
        self.val_dataloader = DataLoader(
            self.val_set,
            batch_size=int(self.exp_cfg.global_batch_size // dist.get_world_size()),
            shuffle=False,
            sampler=None,
            num_workers=dist.get_world_size(),
            pin_memory=True,
            drop_last=False,
            collate_fn=collator
        )

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

    def train_forward(self, tensor, plddts, triplets):
        # apply gaussian noise to the input
        coord_noise = torch.randn_like(tensor.coordinates) * 0.1
        tensor.coordinates = tensor.coordinates + coord_noise

        # mask the input
        masked_tensor, mask = tu.random_mask_and_drop(deepcopy(tensor), triplets)

        # mask plddts   
        plddts[mask['coordinates']] = 0.    # (B, L)
        # average plddt over non-zero values
        avg_plddt = torch.where((plddts != 0).sum(1) > 0, plddts.sum(1) / (plddts != 0).sum(1), 0.) # (B,)
        avg_plddt = avg_plddt.unsqueeze(1).expand(-1, plddts.size(1)) # (B, L)

        out = self.model(
            sequence_tokens=masked_tensor.sequence,
            structure_tokens=masked_tensor.structure,
            average_plddt=avg_plddt,
            per_res_plddt=plddts,
            structure_coords=masked_tensor.coordinates,
            ss8_tokens=masked_tensor.secondary_structure,
        )

        # seq_loss = F.cross_entropy(out.sequence_logits[mask['sequence']], tensor.sequence[mask['sequence']])
        structure_loss = F.cross_entropy(out.structure_logits[mask['structure']], tensor.structure[mask['structure']])
        # loss = 1. * seq_loss + 0.5 * structure_loss
        return structure_loss
    
    def train(self):

        # Variables for monitoring/logging purposes:
        train_steps = self.start_step
        log_steps = 0
        running_loss = 0
        start_time = time()

        self.logger.info(f"Training for {self.exp_cfg.epochs} epochs...")
        self.logger.info(f'effective batch size: {self.exp_cfg.global_batch_size * self.exp_cfg.accumulate_steps}')
        for epoch in range(self.exp_cfg.epochs):
            self.sampler.set_epoch(epoch)

            for step, batch in enumerate(self.train_dataloader):       
                with torch.no_grad():
                    tensor, plddts, triplets = encode_protein(self.model, batch)
                tensor.structure = tensor.structure.detach()
                self.model.train()

                if step % self.exp_cfg.accumulate_steps != 0:
                    with self.model.no_sync():
                        with torch.cuda.amp.autocast(enabled=self.mixed_precision, dtype=torch.bfloat16):
                            loss = self.train_forward(tensor, plddts, triplets)
                        loss = loss / self.exp_cfg.accumulate_steps
                        self.scaler.scale(loss).backward()

                else:
                    with torch.cuda.amp.autocast(enabled=self.mixed_precision, dtype=torch.bfloat16):
                        loss = self.train_forward(tensor, plddts, triplets)
                    loss = loss / self.exp_cfg.accumulate_steps
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.exp_cfg.max_grad_norm)
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    self.optim.zero_grad()

                torch.cuda.empty_cache()
                train_steps += 1
                log_steps += 1
                running_loss += loss.item()*self.exp_cfg.accumulate_steps
                if train_steps % self.exp_cfg.log_interval == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=self.device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    self.logger.info(f"[epoch={(epoch+1):03d}, step={train_steps:07d}] Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    # Log to wandb:
                    if self.exp_cfg.use_wandb and self.rank == 0:
                        wandb.log({
                            'train_loss': avg_loss,
                            'train_steps_per_sec': steps_per_sec,
                        }, step=train_steps)
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()

                # Save checkpoint:
                if (train_steps % self.exp_cfg.save_interval == 0 and train_steps > 0):
                    if self.rank == 0:
                        if self.model_cfg.lora.enabled:
                            self.save_lora(train_steps)
                        else:
                            self.save(train_steps)
                # Evaluate:
                if (train_steps % self.exp_cfg.eval_interval == 0 and train_steps > 0):
                    if self.rank == 0:
                        self.evaluate(train_steps)

            self.scheduler.step()
        
        if self.rank == 0:
            self.logger.info(f"Training completed. Final step: {train_steps}.")
            if self.model_cfg.lora.enabled:
                self.save_lora(train_steps)
            else:
                self.save(train_steps)
    
    def evaluate(self, train_step):
        self.model.eval()
        self.logger.info(f"Evaluating at step {train_step}...")

        # Variables for monitoring/logging purposes:
        running_loss = 0
        log_steps = 0
        start_time = time()

        with torch.no_grad():
            for step, batch in enumerate(self.val_dataloader):
                tensor, plddts, triplets = encode_protein(self.model, batch)
                tensor.structure = tensor.structure.detach()

                loss = self.train_forward(tensor, plddts, triplets)
                running_loss += loss.item()
                log_steps += 1
                
        avg_loss = running_loss / log_steps
        self.logger.info(f"[step={train_step:07d}] Val Loss: {avg_loss:.4f} Time: {time() - start_time:.2f}s")

        if self.exp_cfg.use_wandb:
            wandb.log({
                'val_loss': avg_loss,
            }, step=train_step)
        

    def cleanup(self):
        """
        End DDP training.
        """
        dist.destroy_process_group()


@hydra.main(config_path='esmknot/config', config_name='mlm_tuning', version_base=None)
def main(cfg):
    trainer = ESM3Trainer(cfg)
    trainer.train()
    trainer.cleanup()


if __name__ == "__main__":
    main()
