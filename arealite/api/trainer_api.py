# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import abc
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch.distributed as dist
from datasets import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import TrainerConfig, TrainingArgs
from realhf.base import constants

if TYPE_CHECKING:
    from arealite.system.rollout_controller import RolloutController

# 4. use huggingface.trainerstate
# TODO: how to do checkpointing?

# follow the signature of transformers.Trainer if possible


class Trainer(abc.ABC):
    def __init__(
        self,
        args: TrainingArgs,
        trainer_config: TrainerConfig,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset] = None,
        rollout_controller: Optional["RolloutController"] = None,
    ):
        self.args = args
        self.trainer_config = trainer_config

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.rollout_controller = rollout_controller

        self.train_dataloader = None
        self.valid_dataloader = None

    def create_train_dataloader(self):
        cfg = self.args.train_dataset
        if dist.is_initialized():
            batch_size = cfg.batch_size // dist.get_world_size()
        else:
            batch_size = cfg.batch_size
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=cfg.shuffle,
            pin_memory=cfg.pin_memory,
            num_workers=cfg.num_workers,
            drop_last=True,
            collate_fn=lambda x: x,
        )

    def create_valid_dataloader(self):
        if self.args.valid_dataset is None:
            return
        cfg = self.args.valid_dataset
        if dist.is_initialized():
            batch_size = cfg.batch_size // dist.get_world_size()
        else:
            batch_size = cfg.batch_size
        self.valid_dataloader = StatefulDataLoader(
            dataset=self.valid_dataset,
            batch_size=batch_size,
            shuffle=cfg.shuffle,
            pin_memory=cfg.pin_memory,
            num_workers=cfg.num_workers,
            drop_last=True,
            collate_fn=lambda x: x,
        )

    @property
    def local_train_batch_size(self):
        if not dist.is_initialized():
            return self.args.train_dataset.batch_size
        return self.args.train_dataset.batch_size // dist.get_world_size()

    # TODO: check HF trainer signature
    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None):
        raise NotImplementedError()

    def get_save_checkpoint_path(
        self, epoch: int, step: int, globalstep: int, name: str = "model"
    ):
        path = os.path.join(
            constants.get_save_path(self.args),
            name,
            f"epoch{epoch}epochstep{step}globalstep{globalstep}",
        )
        os.makedirs(path, exist_ok=True)
        return path


@dataclass
class TrainerFactory:
    args: TrainingArgs

    def make_trainer(
        self,
        config: TrainerConfig,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset] = None,
        rollout_controller: Optional["RolloutController"] = None,
    ) -> Trainer:
        if config.type == "grpo":
            from arealite.impl.trainer.grpo import SpmdGRPOTrainer

            return SpmdGRPOTrainer(
                self.args,
                config,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                rollout_controller=rollout_controller,
            )
        elif config.type == "sft":
            from arealite.impl.trainer.sft import SFTTrainer

            return SFTTrainer(
                self.args,
                config,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                rollout_controller=rollout_controller,
            )
        else:
            raise NotImplementedError(f"Unknown trainer type: {config.type}")
