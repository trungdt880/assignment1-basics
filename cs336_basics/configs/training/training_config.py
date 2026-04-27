from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    # Output
    output_dir: str = "./outputs"
    experiment_name: str | None = None

    # Basic training
    max_steps: int = 30000
    max_learnt_tokens: int | None = None  # this will override max_steps
    global_batch_size: int = 1024
    batch_size: int | None = None
    gradient_accumulation_steps: int = 1

    # Optimization
    learning_rate: float = 1e-3
    warmup_learning_rate_ratio: float = 0.1
    betas: tuple[float, float] = (0.9, 0.999)
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.05
    warmup_steps: int = 0  # this will override warmup_ratio
    max_grad_norm: float = 1.0

    # Optimizer choice
    optim: str = "adamw"

    start_from_checkpoint: str | None = None

    # Mixed precision
    tf32: bool = True
    fp16: bool = False
    bf16: bool = True
    eval_bf16: bool = True

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 1000
    save_total_limit: int = 5

    # Evaluation
    eval_strategy: str = "no"  # no, steps, epoch
    eval_steps: int = 500
    eval_batch_size: int = 2
    eval_num_batches: int | None = None
    save_best_eval_metric_name: str = ""
    save_best_eval_metric_greater_is_beter: bool = True

    # DDP
    use_ddp: bool = False
    ddp_bucket_cap_mb: int = 100

    # HW
    num_gpus: int = 1
    dataloader_num_workers: int = 2

    # Experiment tracking
    use_wandb: bool = True
    wandb_project: str = "cs336_hw1"
