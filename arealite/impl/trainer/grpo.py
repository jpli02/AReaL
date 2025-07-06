# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import functools
import os
import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from datasets import Dataset

from arealite import ppo_functional
from arealite.api.cli_args import (
    GRPOTrainerConfig,
    MicroBatchSpec,
    TrainerConfig,
    TrainingArgs,
)
from arealite.api.engine_api import EngineFactory
from arealite.api.io_struct import FinetuneSpec, Trajectory
from arealite.api.llm_client_api import LLMClientFactory
from arealite.api.trainer_api import Trainer
from arealite.system.rollout_controller import RolloutController
from arealite.utils import (
    calc_entropy,
    close_wandb_tensorboard,
    compute_varlen_position_indices,
    concat_padded_tensors,
    gather_logprobs,
    init_stats_logging,
    log_wandb_tensorboard,
    masked_normalization,
    record_timing,
    split_dict_tensor_with_cu_seqlens,
    to_device,
    unpad_input,
)
from realhf.api.core.data_api import load_hf_tokenizer, tabulate_stats
from realhf.base import constants, logging, name_resolve, names, stats_tracker, timeutil

logger = logging.getLogger("GRPO Trainer", "system")


class SpmdGRPOTrainer(Trainer):

    def __init__(
        self,
        args: TrainingArgs,
        trainer_config: TrainerConfig,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset] = None,
        rollout_controller: Optional[RolloutController] = None,
    ):
        super().__init__(
            args,
            trainer_config,
            train_dataset,
            valid_dataset,
            rollout_controller,
        )
        if self.rollout_controller is None:
            raise ValueError("GRPO Trainer requires a rollout controller.")

        assert trainer_config.grpo is not None
        self.config: GRPOTrainerConfig = trainer_config.grpo
        assert args.rollout is not None
        assert self.config.actor is not None

        # Create actor model
        engine_factory = EngineFactory(args)
        self.actor = engine_factory.make_engine(self.config.actor)

        self.actor_tokenizer = load_hf_tokenizer(self.config.actor.path)
        self.gconfig = args.rollout.gconfig

        # Create reference model is specified
        self.ref = None
        if self.config.ref is not None:
            self.ref = engine_factory.make_engine(self.config.ref)

        # Create a client to generate responses and update weights
        client_factory = LLMClientFactory(args)
        self.llm_client = client_factory.make_client(args.rollout.llm_client)

        # Algorithm related attributes
        self.kl_ctl = self.config.kl_ctl
        self.discount = self.config.discount
        self.gae_lambda = self.config.gae_lambda
        self.adv_norm = self.config.adv_norm
        self.max_reward_clip = self.config.max_reward_clip
        self.group_adv_norm = self.config.group_adv_norm
        self.group_size = args.rollout.gconfig.n_samples
        self.max_head_offpolicyness = args.rollout.max_head_offpolicyness
        self.reward_bias = self.config.reward_bias
        self.reward_scaling = self.config.reward_scaling
        self.max_reward_clip = self.config.max_reward_clip

        self.save_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=self.args.exp_ctrl.save_freq_epochs,
            freq_step=self.args.exp_ctrl.save_freq_steps,
            freq_sec=self.args.exp_ctrl.save_freq_secs,
        )
        self.eval_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=self.args.exp_ctrl.eval_freq_epochs,
            freq_step=self.args.exp_ctrl.eval_freq_steps,
            freq_sec=self.args.exp_ctrl.eval_freq_steps,
        )
        self.summary_writer = init_stats_logging(args)

    def train(self, resume_from_checkpoint=None):
        # TODO: handle recover
        self.create_train_dataloader()
        assert self.rollout_controller is not None
        assert self.train_dataloader is not None

        total_epochs = self.args.exp_ctrl.total_train_epochs
        steps_per_epoch = len(self.train_dataloader)
        ft_spec = FinetuneSpec(
            total_train_epochs=total_epochs,
            dataset_size=len(self.train_dataset),
            train_batch_size=self.args.train_dataset.batch_size,
        )

        # Setting up models.
        self.actor.init_distributed(None, ft_spec)
        self.actor.load_model_from_hf(self.config.actor.path)
        self.actor.eval()
        if self.ref is not None:
            self.ref.init_distributed(None, ft_spec)
            self.ref.load_model_from_hf(self.config.ref.path)
            self.ref.eval()
        self.llm_client.wait_until_servers_ready()
        self.actor.update_weights_to(self.llm_client)

        # Start rollout for asynchronous RL.
        if self.config.async_training:
            self.rollout_controller.start_generate_loop()

        # Main RL training loop.
        total_epochs = self.args.exp_ctrl.total_train_epochs
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        global_step = 0
        warmup_steps = self.max_head_offpolicyness + 1
        assert steps_per_epoch >= warmup_steps
        start_time = time.monotonic()
        for epoch in range(total_epochs):
            for step, data in enumerate(self.train_dataloader):
                timing_stats = {}
                with record_timing("timeperf/rollout", timing_stats):
                    if self.config.async_training:
                        self.rollout_controller.submit(data)
                        # Submitted data will not actually be sent for rollout.
                        # The rollout controller over-subscribe the data to
                        # ensure that there are enough data being generated.
                        if epoch == 0 and step < warmup_steps:
                            continue
                        # Wait until enough trajectories has been collected.
                        trajs = self.rollout_controller.prepare_batch(
                            batch_size=self.args.train_dataset.batch_size // world_size
                        )
                    else:
                        # Run batched rollout by submitting requests to LLM servers
                        trajs = self.rollout_controller.generate_batch(
                            batch_size=len(data),
                            env_options=data,
                        )

                with record_timing("timeperf/train_step", timing_stats):
                    # Run RL training and update weights.
                    mb_stats = self._train_step(trajs)
                    self.actor.step_lr_scheduler()

                with record_timing("timeperf/sync_weights", timing_stats):
                    # Synchronize weights to the client.
                    self.actor.update_weights_to(self.llm_client)
                    # Update model version
                    name = names.model_version(
                        self.args.experiment_name, self.args.trial_name, "actor"
                    )
                    name_resolve.add(name, str(global_step + 1), replace=True)

                if self.save_ctl.check(
                    epochs=int(step == steps_per_epoch - 1), steps=1
                ):
                    if dist.get_rank() == 0:
                        logger.info("Saving model ...")
                    with record_timing("timeperf/save", timing_stats):
                        save_path = os.path.join(
                            constants.get_save_path(self.args), "actor"
                        )
                        self.actor.save_model_to_hf(
                            save_path,
                            tokenizer=self.actor_tokenizer,
                            base_model_path=self.config.actor.path,
                        )

                assert len(mb_stats) == self.config.ppo_n_minibatches
                log_step = self.config.ppo_n_minibatches * global_step
                for i, stats in enumerate(mb_stats):
                    log_wandb_tensorboard(log_step + i, stats, self.summary_writer)
                log_wandb_tensorboard(log_step, timing_stats, self.summary_writer)

                if dist.get_rank() == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{total_epochs} "
                        f"Step {step+1}/{steps_per_epoch} "
                        f"Train step {global_step + 1}/{total_epochs * steps_per_epoch - warmup_steps} done."
                    )
                    logger.info(
                        f"Detailed time stats: \n{tabulate_stats(timing_stats, floatfmt='.2f')}"
                    )
                    for i, stats in enumerate(mb_stats):
                        logger.info(
                            f"GRPO training stats ({i + 1}/{len(mb_stats)}):\n{tabulate_stats(stats)}"
                        )

                global_step += 1

        if dist.get_rank() == 0:
            logger.info(
                f"Training completes! Total time elapsed {time.monotonic() - start_time:.2f}."
            )
        if self.config.async_training:
            self.rollout_controller.stop_generate_loop()

        close_wandb_tensorboard(self.summary_writer)

    def _train_step(self, trajs: List[Trajectory]):
        rollout = concat_padded_tensors([traj.data for traj in trajs])
        rollout = to_device(rollout, torch.cuda.current_device())

        # Marks which sequence does not has an EOS token, i.e.,
        # generation is truncated by the configured maximum generation length
        batch_tokens = rollout["input_ids"]
        seq_no_eos_mask = (
            batch_tokens[:, -1] != self.actor_tokenizer.eos_token_id
        ).logical_and(batch_tokens[:, -1] != self.actor_tokenizer.pad_token_id)

        # Remove padding to use flash-attn
        attn_mask = rollout["attention_mask"]
        input_ids, _, cu_seqlens, max_seqlen = unpad_input(
            rollout["input_ids"], attn_mask
        )
        position_ids = compute_varlen_position_indices(input_ids.shape[0], cu_seqlens)

        # Transformer forward input data
        model_inputs = dict(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=None,
            position_ids=position_ids.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            use_cache=False,
        )
        old_logp, *_ = unpad_input(rollout["logprobs"], attn_mask)
        prompt_mask, *_ = unpad_input(rollout["prompt_mask"], attn_mask)
        # Shift logprobs and mask for computing loss.
        loss_mask = prompt_mask.logical_not()
        loss_mask = torch.roll(loss_mask, shifts=-1)
        old_logp = torch.roll(old_logp, shifts=-1)

        input_ids = model_inputs["input_ids"].squeeze(0)
        n_seqs = seq_no_eos_mask.shape[0]
        assert n_seqs == self.local_train_batch_size * self.group_size, (
            n_seqs,
            self.group_size,
            self.local_train_batch_size,
        )

        # Run reference model forward
        def calc_logprobs(logits, input_data):
            logits = logits.squeeze(0).float()
            labels = torch.roll(input_data["input_ids"].squeeze(0), shifts=-1)
            logits /= self.gconfig.temperature
            logprobs = gather_logprobs(logits, labels)
            return logprobs.unsqueeze(0)

        if self.ref is not None and self.config.kl_ctl != 0.0:
            ref_logp = self.ref.forward(
                model_inputs,
                mb_spec=self.config.mb_spec,
                post_hook=calc_logprobs,
            ).squeeze(0)
        else:
            ref_logp = torch.zeros_like(input_ids, dtype=torch.float32)

        # Recompute logprobs using the current actor model.
        prox_logp = None
        if self.config.recompute_logprob:
            _logp = self.actor.forward(
                model_inputs,
                mb_spec=self.config.mb_spec,
                post_hook=calc_logprobs,
            ).squeeze(0)
            if self.config.use_decoupled_loss:
                prox_logp = _logp
            else:
                # Overwrite the logp returned by the inference engine
                old_logp = _logp

        # Compute rewards using the reward function in synchronous RLVR pipeline.
        reward_score = rollout["rewards"]
        reward_score = (reward_score + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(reward_score, max=self.max_reward_clip)
        if self.config.group_reward_norm:
            for i in range(n_seqs // self.group_size):
                s = slice(i * self.group_size, (i + 1) * self.group_size)
                r = reward_score[s]
                reward_score[s] = (r - r.mean()) / (r.std() + 1e-9)

        # Apply the mask to log probabilities.
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute KL-regularized rewards and GAEs.
        cu_seqlens = model_inputs["cu_seqlens"]
        seq_no_eos_mask = seq_no_eos_mask
        kl_rewards, rewards = ppo_functional.get_packed_rewards(
            kl_ctl=self.kl_ctl,
            clip_reward_value=self.max_reward_clip,
            log_probs=old_logp,
            ref_log_probs=ref_logp,
            reward_score=reward_score,
            cu_seqlens=cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
            mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
        )
        advantages, _ = ppo_functional.get_packed_advantages_and_returns(
            gamma=self.discount,
            lam=self.gae_lambda,
            values=torch.zeros(
                input_ids.shape[0] + n_seqs,
                device=input_ids.device,
                dtype=torch.float32,
            ),
            rewards=rewards,
            short1cu_seqlens=cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )

        # Optionally perform advantage normalization.
        if self.adv_norm:
            if self.group_adv_norm:
                n_samples = len(cu_seqlens) - 1
                assert n_samples % self.group_size == 0
                adv_list = []
                for i in range(0, n_samples, self.group_size):
                    adv_list.append(
                        masked_normalization(
                            advantages[cu_seqlens[i] : cu_seqlens[i + self.group_size]],
                            loss_mask[cu_seqlens[i] : cu_seqlens[i + self.group_size]],
                            all_reduce=False,
                        )
                    )
                advantages = torch.cat(adv_list, 0)
            else:
                advantages = masked_normalization(advantages, loss_mask)

        # Prepare data to be splitted into mini-batches.
        global_batch = dict(
            **model_inputs,
            old_logp=old_logp,
            advantages=advantages,
            loss_mask=loss_mask,
            prox_logp=prox_logp,
        )
        input_lens = model_inputs["cu_seqlens"][1:] - model_inputs["cu_seqlens"][:-1]

        all_stats = []
        with stats_tracker.scope("actor"):
            ########## Logging code starts ##########
            result_denominators = {
                "correct_n_seqs": (reward_score > 0).bool(),
                "incorrect_n_seqs": (reward_score <= 0).bool(),
            }
            global_denominators = dict(
                n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
                n_tokens=torch.ones_like(loss_mask, dtype=torch.bool),
                n_valid_tokens=loss_mask.bool(),
                **result_denominators,
            )
            stats_tracker.denominator(**global_denominators)
            stats_tracker.stat(
                correct_seq_len=input_lens.float(), denominator="correct_n_seqs"
            )
            stats_tracker.stat(
                incorrect_seq_len=input_lens.float(), denominator="incorrect_n_seqs"
            )

            stats = dict(
                advantages=advantages,
                kl_rewards=kl_rewards,
                final_reward=rewards,
            )
            stats_tracker.stat(**stats, denominator="n_valid_tokens")

            prompt_lens = []
            for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:]):
                prompt_lens.append(prompt_mask[s:e].sum())
            prompt_lens = torch.tensor(prompt_lens, device=reward_score.device)
            seq_stats = dict(
                no_eos_ratios=seq_no_eos_mask.float(),
                task_reward=reward_score,
                prompt_len=prompt_lens.float(),
                seq_len=input_lens.float(),
            )
            stats_tracker.stat(**seq_stats, denominator="n_seqs")
            scalars = dict(
                mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
                eps_clip=self.config.eps_clip,
                use_prox_logp=prox_logp is not None,
            )
            if self.config.c_clip is not None:
                scalars["c_clip"] = self.config.c_clip
                scalars["use_dual_clip"] = 1
            else:
                scalars["use_dual_clip"] = 0
            if self.config.behav_imp_weight_cap is not None:
                scalars["behav_imp_weight_cap"] = self.config.behav_imp_weight_cap
            stats_tracker.scalar(**scalars)

            global_stats = stats_tracker.export()
            for k in global_denominators:
                global_stats.pop(f"actor/{k}")
            ########## Logging code ends ##########

            mb_inputs = split_dict_tensor_with_cu_seqlens(
                global_batch,
                mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
            )
            for mb in mb_inputs.mbs:
                model_inputs = {k: mb[k] for k in model_inputs}
                train_stat = self.actor.train_batch(
                    mb,
                    loss_fn=functools.partial(
                        grpo_loss_fn,
                        temperature=self.gconfig.temperature,
                        eps_clip=self.config.eps_clip,
                        c_clip=self.config.c_clip,
                        behav_imp_weight_cap=self.config.behav_imp_weight_cap,
                    ),
                    mb_spec=self.config.mb_spec,
                    loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
                )
                stats_tracker.scalar(**train_stat)
                all_stats.append(stats_tracker.export())
        all_stats[0].update(global_stats)
        return all_stats


def grpo_loss_fn(
    logits: torch.Tensor,
    input_data: Dict,
    temperature: float,
    eps_clip: float,
    c_clip: float | None,
    behav_imp_weight_cap: float | None,
):
    """Loss function for actor step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    input_ids = input_data["input_ids"].squeeze(0)
    cu_seqlens = input_data["cu_seqlens"]
    old_logp = input_data["old_logp"]
    advantages = input_data["advantages"]
    loss_mask = input_data["loss_mask"]
    prox_logp = input_data["prox_logp"]

    logits = logits.squeeze(0).float()
    logits /= temperature
    logprobs = gather_logprobs(logits, torch.roll(input_ids, shifts=-1))
    loss, stat = ppo_functional.actor_loss_fn(
        logprobs=logprobs,
        old_logprobs=old_logp,
        advantages=advantages,
        eps_clip=eps_clip,
        loss_mask=loss_mask,
        c_clip=c_clip,
        proximal_logprobs=prox_logp,
        behav_imp_weight_cap=behav_imp_weight_cap,
    )

    entropy = calc_entropy(logits=logits, cu_seqlens=cu_seqlens)

    # Log training statistics
    stats_tracker.denominator(
        n_tokens=torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device),
        n_valid_tokens=loss_mask.bool(),
        clipped_tokens=stat["clip_mask"],
        dual_clipped_tokens=stat["dual_clip_mask"],
    )

    stats_tracker.stat(
        importance_weight=stat["importance_weight"],
        approx_kl=stat["approx_kl"],
        new_logp=logprobs.detach(),
        old_logp=old_logp,
        entropy=entropy.float(),
        actor_loss=stat["loss"],
        clip_ratio=stat["clip_mask"].float(),
        dual_clip_ratio=stat["dual_clip_mask"].float(),
        denominator="n_valid_tokens",
    )
    if "behave_imp_weight" in stat:
        stats_tracker.denominator(unclipped_behave_tokens=stat["behave_mask"])
        stats_tracker.stat(
            behave_imp_weight=stat["behave_imp_weight"],
            behave_approx_kl=stat["behave_approx_kl"],
            denominator="unclipped_behave_tokens",
        )
    vocab_min_logits = logits.detach().min(-1).values.float()
    vocab_max_logits = logits.detach().max(-1).values.float()
    stats_tracker.stat(
        vocab_min_logits=vocab_min_logits,
        vocab_max_logits=vocab_max_logits,
        denominator="n_tokens",
    )

    clip_mask = stat["clip_mask"]
    clipped_new_logp = torch.where(clip_mask, logprobs.detach(), 0.0)
    clipped_old_logp = torch.where(clip_mask, old_logp, 0.0)
    stats_tracker.stat(
        clipped_new_logp=clipped_new_logp,
        clipped_old_logp=clipped_old_logp,
        denominator="clipped_tokens",
    )
    return loss
