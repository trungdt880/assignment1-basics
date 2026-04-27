import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import tyro
from loguru import logger
from omegaconf import OmegaConf
from tqdm import trange

import wandb
from cs336_basics.configs.base_config import Config, get_default_config
from cs336_basics.models.common import (
    cross_entropy_loss,
    get_batch,
    gradient_clipping,
    lr_cosine_schedule,
    save_checkpoint,
    set_seed,
)
from cs336_basics.models.optimizers import AdamW
from cs336_basics.models.transformer import TransformerLM


def main(config: Config):
    config.validate()

    set_seed(config.data.seed)

    if config.training.experiment_name is None:
        output_dir = Path(config.training.output_dir)
        experiment_name = output_dir.name
    else:
        output_dir = Path(config.training.output_dir) / config.training.experiment_name
        experiment_name = config.training.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_cfg_dir = output_dir / "experiment_cfg"

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    wandb_config_file = output_dir / "wandb_config.json"
    with open(wandb_config_file, "w") as f:
        json.dump(
            {"project": config.training.wandb_project, "run_id": experiment_name}, f
        )
    logger.info(f"Saved config to {save_cfg_dir}")

    if config.training.use_wandb:
        config_dict = {
            **config.__dict__,
        }
        wandb.init(
            project=config.training.wandb_project,
            name=experiment_name,
            config=config_dict,
        )

    model = TransformerLM(**config.model.__dict__)

    train_data = np.load(config.data.dataset_path, mmap_mode="r")
    val_data = np.load(config.data.val_dataset_path, mmap_mode="r")
    eval_bs = config.training.eval_batch_size
    eval_batch_length = eval_bs * config.model.context_length
    max_eval_batches = (len(val_data) - 1) // eval_batch_length
    assert max_eval_batches > 0, f"Val dataset is too small: {len(val_data)}"
    if config.training.eval_num_batches is None:
        num_eval_batches = max_eval_batches
    else:
        num_eval_batches = min(config.training.eval_num_batches, max_eval_batches)

    optimizer = AdamW(
        model.parameters(),
        config.training.learning_rate,
        config.training.betas,
        config.training.weight_decay,
    )

    if config.training.num_gpus > 0:
        assert torch.cuda.is_available(), f"GPU is not detected."
        device = "cuda"
    else:
        device = "cpu"

    model.to(device)
    model.train()

    max_steps = config.training.max_steps
    if config.training.max_learnt_tokens is not None:
        max_steps = int(
            config.training.max_learnt_tokens
            / config.training.global_batch_size
            / config.model.context_length
        )
    config.training.max_steps = max_steps

    warmup_lr = (
        config.training.learning_rate * config.training.warmup_learning_rate_ratio
    )
    warmup_steps = int(config.training.warmup_ratio * max_steps)

    config.save(save_cfg_dir / "config.yaml")
    omegaconf_config = OmegaConf.create(config.__dict__)
    omegaconf_config["max_steps"] = max_steps
    omegaconf_config["save_steps"] = config.training.save_steps
    omegaconf_config["warmup_lr"] = warmup_lr
    omegaconf_config["warmup_steps"] = warmup_steps
    omegaconf_config["num_eval_batches"] = num_eval_batches
    OmegaConf.save(omegaconf_config, save_cfg_dir / "conf.yaml", resolve=True)

    for i in trange(max_steps, desc="Steps"):
        x, y = get_batch(
            train_data,
            config.training.global_batch_size,
            config.model.context_length,
            device,
        )

        pred_y = model(x)
        loss = cross_entropy_loss(pred_y, y)

        loss.backward()
        gradient_clipping(model.parameters(), config.training.max_grad_norm)

        lr = lr_cosine_schedule(
            i,
            warmup_lr,
            config.training.learning_rate,
            warmup_steps,
            max_steps,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if (i % config.training.save_steps == 0 and i > 0) or i == max_steps - 1:
            model_outpath = ckpt_dir / f"{str(i).zfill(9)}.pt"
            save_checkpoint(model, optimizer, i, model_outpath.resolve().as_posix())

        log_dict = dict(loss=loss.detach().item())
        if (i % config.training.eval_steps == 0 and i > 0) or i == max_steps - 1:
            with torch.no_grad():
                model.eval()
                val_loss = 0
                for val_idx in trange(num_eval_batches, desc="Eval"):
                    x = val_data[
                        val_idx * eval_batch_length : (val_idx + 1) * eval_batch_length
                    ].reshape(eval_bs, config.model.context_length)
                    y = val_data[
                        val_idx * eval_batch_length
                        + 1 : (val_idx + 1) * eval_batch_length
                        + 1
                    ].reshape(eval_bs, config.model.context_length)
                    x = torch.tensor(x, dtype=torch.long, device=device)
                    y = torch.tensor(y, dtype=torch.long, device=device)
                    pred_y = model(x)
                    val_loss += cross_entropy_loss(pred_y, y).item()
                val_loss /= num_eval_batches
                val_ppl = math.exp(val_loss)
                val_bpt = val_loss / math.log(2)
            model.train()
            log_dict.update(
                dict(
                    val_loss=val_loss,
                    val_ppl=val_ppl,
                    val_bits_per_token=val_bpt,
                )
            )

        if config.training.use_wandb:
            wandb.log(log_dict, step=i)


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    config = tyro.cli(Config, default=get_default_config(), description=__doc__)

    if config.load_config_path:
        p = Path(config.load_config_path)
        assert p.exists(), f"Config path doesn't exist: {p}"
        config = config.load(p)
        config.load_config_path = None
        logger.info(f"Loaded config: {p}")

        # Override with command-line
        config = tyro.cli(Config, default=config, description=__doc__)

    main(config)
